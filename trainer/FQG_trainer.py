# -*- coding: utf-8 -*-
import os
import shutil
import json
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
# from sklearn.metrics import accuracy_score
from .config import *
from util.file_utils import load
from util.tensor_utils import to_thinnest_padded_tensor
# from util.tensor_utils import transform_tensor_by_dict
from modules.ema import EMA
from metric.text_generation_metrics import compute_metrics_by_list, calc_style_acc
from loss.text_generation_losses import NMTLoss, QGLoss, SoftQGLoss1, SoftQGLoss2


class Trainer(object):

    def __init__(self, args, model, train_dataloader, dev_dataloader,
                 loss, optimizer, scheduler, device, emb_dicts=None,
                 logger=None, partial_models=None, partial_resumes=None,
                 partial_trainables=None):
        self.args = args
        self.device = device
        self.logger = logger
        self.dicts = emb_dicts

        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.val_num_batches = args.val_num_examples // args.batch_size

        self.model = model
        self.identifier = type(model).__name__ + '_'

        self.loss = loss
        self.nmt_loss = NMTLoss(self.model.predict_size)  # !!! !!!!!!!!!!!!!!!!!!! parallel will cause error, no attribute predict_size
        self.copy_loss = nn.NLLLoss(reduction="sum")  # !!!
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = None
        if args.use_ema:
            self.ema = EMA(args.ema_decay)
            self.ema.register(model)

        # VARIABLE
        self.train_eval_dict = None
        self.dev_eval_dict = None
        self.totalBatchCount = 0
        self.best_result = {
            "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0,
            "rougeL": 0.0, "meteor": 0.0}
        self.result_keys = list(self.best_result.keys())

        self.start_time = datetime.now().strftime('%b-%d_%H-%M')
        self.start_epoch = 1
        self.step = 0
        if args.resume:
            self._resume_checkpoint(args.resume)
            self.model = self.model.to(self.device)
            for state in self.optimizer.optimizer.state.values():  # !!!
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        if args.resume_partial:
            num_partial_models = len(partial_models)
            for i in range(num_partial_models):
                self._resume_model(
                    partial_resumes[i],
                    partial_models[i], partial_trainables[i])

    def _update_best_result(self, new_result, best_result):
        is_best = False
        # VARIABLE
        if (new_result["bleu4"] > best_result["bleu4"]):
            is_best = True
        for key in self.result_keys:
            best_result[key] = max(best_result[key], new_result[key])
        return best_result, is_best

    def _result2string(self, result, result_keys):
        string = ""
        for key in result_keys:
            string += "_" + key + "_" + ("{:.5f}").format(result[key])
        return string

    def train(self):
        patience = 0
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            result = self._train_epoch(epoch)

            self.best_result, is_best = self._update_best_result(
                result, self.best_result)

            if self.args.use_early_stop:
                if (not is_best):
                    patience += 1
                    if patience > self.args.early_stop:
                        print("Perform early stop!")
                        break
                else:
                    patience = 0

            if epoch % self.args.save_freq == 0:
                self._save_checkpoint(
                    epoch, result, self.result_keys, is_best)  # !!!
        return self.model

    def eval(self, dataloader, eval_file, output_file):
        eval_dict = load(eval_file)
        result = self._valid(eval_dict, dataloader)
        print("eval: " + self._result2string(result, self.result_keys))
        if output_file is not None:
            with open(output_file, 'w', encoding='utf8') as outfile:
                json.dump(result, outfile)
        return result

    def test(self, dataloader, output_file):
        result, example_sid, example_pid, example_ans_sent, example_answer_text, example_char_start, example_char_end = self._test(dataloader)
        if output_file is not None:
            with open(output_file, 'w', encoding='utf8') as outfile:
                for i in range(len(result)):
                    q = result[i].rstrip()
                    to_print = [
                        str(example_pid[i]),
                        str(example_sid[i]),
                        q,
                        str(example_ans_sent[i]),
                        str(example_answer_text[i]),
                        str(example_char_start[i]),
                        str(example_char_end[i])]

                    outfile.write("\t".join(to_print).rstrip().replace("\n", "\\n") + "\n")
        return result

    def _train_epoch(self, epoch):
        self.model.train()
        self.model.to(self.device)

        # initialize
        global_loss = {"total": 0, "qg": 0, "clue": 0, "style": 0}
        last_step = self.step - 1
        last_time = time.time()

        # train over batches
        for batch_idx, batch in enumerate(self.train_dataloader):
            # get batch
            self.totalBatchCount += 1
            for k in batch.keys():
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device)
            # batch_num_src_tokens = (batch["ans_sent_word_ids"] != self.dicts["word"]["<pad>"]).sum().item()
            batch_num_tgt_tokens = (batch["ques_word_ids"] != self.dicts["word"]["<pad>"]).sum().item()

            # calculate loss and back propagation
            self.model.zero_grad()
            predict = self.model(batch)
            # VARIABLE
            if self.args.copy_type in ["soft", "soft-oov"]:
                g_outputs, c_outputs, c_gate_values, src_max_len, soft_g_outputs, new_g_outputs = predict
            else:
                g_outputs, c_outputs, c_gate_values, src_max_len = predict  # y_clue_logits, y_style_logits

            targets, max_len = to_thinnest_padded_tensor(batch["tgt"])
            targets = targets.transpose(0, 1)[1:]  # !!! exclude <s>

            batch_switch = None
            batch_copy_position = None
            if self.args.copy_type in ["hard-oov", "soft-oov"]:
                batch_switch = batch["switch_oov"]
                batch_copy_position = batch["copy_position_oov"]
            elif self.args.copy_type in ["hard", "soft"]:
                batch_switch = batch["switch"]
                batch_copy_position = batch["copy_position"]
            else:
                print("copy type must be hard-oov, hard or soft")

            if self.args.copy_type in ["soft", "soft-oov"]:
                batch_switch_soft = batch["switch_soft"]
                batch_copy_position_soft = batch["copy_position_soft"]

            if self.args.only_copy_content:
                batch_switch = batch_switch * batch["ques_is_content"].float()  # NOTICE
                batch_copy_position = batch_copy_position * batch["ques_is_content"]

            if self.args.copy_type in ["soft", "soft-oov"] and self.args.only_copy_content:
                batch_switch_soft = batch_switch_soft * batch["ques_is_content"].float()  # NOTICE
                batch_copy_position_soft = batch_copy_position_soft * batch["ques_is_content"]

            copy_switch = batch_switch[:, :max_len].transpose(0, 1)[1:]
            c_targets = batch_copy_position[:, :max_len].transpose(0, 1)[1:]

            if self.args.copy_type in ["soft", "soft-oov"]:
                copy_switch_soft = batch_switch_soft[:, :max_len].transpose(0, 1)[1:]
                c_targets_soft = batch_copy_position_soft[:, :max_len].transpose(0, 1)[1:]

            if self.args.copy_type in ["hard", "hard-oov"]:
                loss_qg = QGLoss(
                    g_outputs, targets,
                    c_outputs, copy_switch,
                    c_gate_values, c_targets,
                    self.nmt_loss, self.copy_loss) / batch_num_tgt_tokens
            elif self.args.copy_type in ["soft", "soft-oov"] and self.args.copy_loss_type == 1:# TODO
                loss_qg = SoftQGLoss1(
                    g_outputs, targets,
                    c_outputs, copy_switch,
                    c_gate_values, c_targets,
                    self.nmt_loss, self.copy_loss,
                    soft_g_outputs, new_g_outputs) / batch_num_tgt_tokens
            elif self.args.copy_type in ["soft", "soft-oov"] and self.args.copy_loss_type == 2:# TODO
                loss_qg = SoftQGLoss2(
                    g_outputs, targets,
                    c_outputs, copy_switch, copy_switch_soft,
                    c_gate_values, c_targets, c_targets_soft,
                    self.nmt_loss, self.copy_loss,
                    soft_g_outputs, new_g_outputs) / batch_num_tgt_tokens
            else:
                loss_qg = None
                print("copy_type and copy_loss_type config error!")

            loss = loss_qg

            loss.backward()
            global_loss["total"] += loss.item()
            global_loss["qg"] += loss_qg.item()

            # gradient clip
            if (not self.args.no_grad_clip):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad_norm)

            # update model
            self.optimizer.step()

            # update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # exponential moving avarage
            if self.args.use_ema:
                self.ema(self.model, self.step)

            # print training info
            if self.step % self.args.print_freq == self.args.print_freq - 1:
                used_time = time.time() - last_time
                step_num = self.step - last_step
                speed = self.train_dataloader.batch_size * \
                    step_num / used_time
                batch_loss = {k: v / step_num for k, v in global_loss.items()}
                print(("step: {}/{} \t "
                       "epoch: {} \t "
                       "lr: {} \t "
                       "loss: {} \t "
                       "speed: {} examples/sec").format(
                           batch_idx, len(self.train_dataloader),
                           epoch,
                           #  self.optimizer.param_groups[0]['lr'],  # !!!!!!!! because we used special optim
                           self.optimizer.lr,
                           str(batch_loss),
                           speed))
                global_loss = {k: 0 for k in global_loss}
                last_step = self.step
                last_time = time.time()
            self.step += 1

            if self.args.debug and batch_idx >= self.args.debug_batchnum:
                break

        # evaluate, log, and visualize for each epoch
        train_result = self._valid(self.train_eval_dict, self.train_dataloader)
        dev_result = self._valid(self.dev_eval_dict, self.dev_dataloader)
        self.model.decoder.attn.mask = None
        if self.args.use_answer_separate:
            self.model.decoder.ans_attn.mask = None
        self.optimizer.updateLearningRate(dev_result["bleu4"], epoch)
        print("train_result: " +
              self._result2string(train_result, self.result_keys) +
              "dev_result: " +
              self._result2string(dev_result, self.result_keys))

        return dev_result

    def _valid(self, eval_dict, dataloader):
        if self.args.use_ema:
            self.ema.assign(self.model)
        self.model.eval()

        # VARIABLE
        predict, gold = [], []

        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dataloader)):
                for k in batch.keys():
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.device)
                # ground truth
                # VARIABLE
                tgt_batch = batch["tgt_tokens"]
                (pred_batch, maskS) = \
                    self.model.translate_batch(batch)  # y_clue, y_style

                # nltk BLEU evaluator needs tokenized sentences
                gold += [[r] for r in tgt_batch]
                predict += pred_batch

                # if self.args.use_clue_predict:
                #     gold_clue += torch.masked_select(
                #         batch["ans_sent_is_overlap"],
                #         maskS.byte()).long().tolist()
                #     predict_clue += torch.masked_select(
                #         y_clue, maskS.byte()).long().tolist()
                # if self.args.use_style_predict:
                #     gold_style += batch[
                #         "ques_type_id"].long().squeeze().tolist()
                #     predict_style += y_style.long().squeeze().tolist()

                if((batch_idx + 1) == self.val_num_batches):
                    break

                if self.args.debug and batch_idx >= self.args.debug_batchnum:
                    break

        # get evaluation result by comparing truth and prediction
        # VARIABLE
        no_copy_mark_predict = [[word.replace('[[', '').replace(']]', '')
                                 for word in sent] for sent in predict]

        # calculate result metrics
        other_metrics = compute_metrics_by_list(
            gold, no_copy_mark_predict, f_gold="gold.txt", f_pred="pred.txt")
        clue_acc, style_acc = -1, -1
        # if self.args.use_clue_predict:
        #     clue_acc = accuracy_score(gold_clue, predict_clue)
        # if self.args.use_style_predict:
        #     style_acc = accuracy_score(gold_style, predict_style)
        pred_style_acc = calc_style_acc(gold, no_copy_mark_predict)
        result = {
            "rougeL": other_metrics["ROUGE_L"],
            "meteor": other_metrics["METEOR"],
            "bleu1": other_metrics["Bleu_1"],
            "bleu2": other_metrics["Bleu_2"],
            "bleu3": other_metrics["Bleu_3"],
            "bleu4": other_metrics["Bleu_4"],
            "clue_acc": clue_acc,
            "style_acc": style_acc,
            "pred_style_acc": pred_style_acc}
        print("Valid results: ", result)

        if self.args.use_ema:
            self.ema.resume(self.model)
        self.model.train()
        return result

    def _test(self, dataloader):
        if self.args.use_ema:
            self.ema.assign(self.model)
        self.model.eval()

        # VARIABLE
        predict = []
        example_sid = []
        example_pid = []
        example_ans_sent = []
        example_answer_text = []
        example_char_start = []
        example_char_end = []

        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dataloader)):
                for k in batch.keys():
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.device)
                # ground truth
                # VARIABLE
                (pred_batch, maskS) = \
                    self.model.translate_batch(batch)  # y_clue, y_style

                # nltk BLEU evaluator needs tokenized sentences
                predict += pred_batch
                example_sid += list(batch["sid"])
                example_pid += list(batch["pid"])

                example_ans_sent += list(batch["ans_sent"])
                example_answer_text += list(batch["answer_text"])
                example_char_start += list(batch["char_start"])
                example_char_end += list(batch["char_end"])

                if self.args.debug and batch_idx >= self.args.debug_batchnum:
                    break

        # get evaluation result by comparing truth and prediction
        # VARIABLE
        no_copy_mark_predict = [[word.replace('[[', '').replace(']]', '')
                                 for word in sent] for sent in predict]
        generate_result = [" ".join(token_list) for token_list in no_copy_mark_predict]

        if self.args.use_ema:
            self.ema.resume(self.model)
        self.model.train()
        return generate_result, example_sid, example_pid, example_ans_sent, example_answer_text, example_char_start, example_char_end

    def _save_checkpoint(self, epoch, result, result_keys, is_best):
        if self.args.use_ema:
            self.ema.assign(self.model)
        arch = type(self.model).__name__
        state = {
            'epoch': epoch,
            'arch': arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.optimizer.state_dict(),  # !!!
            'best_result': self.best_result,
            'step': self.step + 1,
            'start_time': self.start_time}
        filename = os.path.join(
            self.args.checkpoint_dir,
            self.identifier +
            'checkpoint_epoch{:02d}'.format(epoch) +
            self._result2string(result, result_keys) + '.pth.tar')
        print("Saving checkpoint: {} ...".format(filename))
        if not os.path.exists(self.args.checkpoint_dir):
            os.makedirs(self.args.checkpoint_dir)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(
                filename, os.path.join(
                    self.args.checkpoint_dir, 'model_best.pth.tar'))
        if self.args.use_ema:
            self.ema.resume(self.model)
        return filename

    def _resume_checkpoint(self, resume_path):
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(
            resume_path, map_location=lambda storage, loc: storage)
        self.start_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.optimizer.load_state_dict(checkpoint['optimizer'])  # !!!
        self.best_result = checkpoint['best_result']
        self.step = checkpoint['step']
        self.start_time = checkpoint['start_time']
        if self.scheduler is not None:
            self.scheduler.last_epoch = checkpoint['epoch']
        print("Checkpoint '{}' (epoch {}) loaded".format(
            resume_path, self.start_epoch))

    def _resume_model(self, resume_path, model, trainable=True):
        checkpoint = torch.load(
            resume_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        if not trainable:
            for p in model.parameters():
                p.requires_grad = False
        print("Model loaded")

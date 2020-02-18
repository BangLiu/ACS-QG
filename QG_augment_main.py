# -*- coding: utf-8 -*-
"""
Factorized question generation main file.
It trains a question generation model based on SQuAD or other datasets.
After training, it can generate questions for given dataset.
"""
import math
import codecs
import torch
import torch.nn as nn
from datetime import datetime
from data_loader.FQG_augment_data import prepro, get_loader
from trainer.FQG_trainer import Trainer
from optimizer.optim import Optim
from util.file_utils import load
from util.exp_utils import set_device, set_random_seed
from util.exp_utils import set_logger, summarize_model, get_checkpoint_dir
from config import *


def get_qa_input_file(qg_result_file, da_paragraphs_file, qa_data_file):
    # load dict (pid, paragra)
    pid_para_dict = {}
    with codecs.open(da_paragraphs_file, "r", encoding='utf8') as fp:
        lines = fp.readlines()
        for line in lines:
            line_split = line.rstrip().split("\t")
            # print("line_split is: ", line_split)
            pid = str(line_split[0])
            para = str(line_split[1])
            pid_para_dict[pid] = para

    # read qg_result_file
    fqa = codecs.open(qa_data_file, "w", encoding='utf8')
    with codecs.open(qg_result_file, "r", encoding='utf8') as fqg:
        lines = fqg.readlines()
        for line in lines:
            line_split = str(line).rstrip().split("\t")
            example_pid = line_split[0]
            example_sid = line_split[1]
            q = line_split[2]
            example_ans_sent = line_split[3]
            example_answer_text = line_split[4]
            example_char_start = line_split[5]
            example_char_end = line_split[6]

            # append paragraph
            example_paragraph = pid_para_dict[str(example_pid)]

            # calculate start and end character position in paragraph
            sentence_start = example_paragraph.find(example_ans_sent)
            if sentence_start < 0:  # if not found, there must be some potential problem
                print("BAD CASE: " + "\t" + example_paragraph + "\t" + example_ans_sent)
                continue
            else:
                p_char_start = str(int(sentence_start) + int(example_char_start))
                p_char_end = str(int(sentence_start) + int(example_char_end))

            # write to qa_data_file
            to_print = [
                example_pid, example_sid, q, example_ans_sent,
                example_answer_text, example_char_start, example_char_end,
                example_paragraph, p_char_start, p_char_end]
            fqa.write("\t".join(to_print) + "\n")
    fqa.close()
    fp.close()


def main(args):
    # import model according to input args
    if args.net == "FQG":
        from model.FQG_model import FQG as Model
    else:
        print("Default use s2s_qanet model.")
        from model.FQG_model import FQG as Model

    # configure according to input args and some experience
    emb_config["word"]["emb_size"] = args.tgt_vocab_limit
    args.emb_config["word"]["emb_size"] = args.tgt_vocab_limit
    args.brnn = True
    args.lower = True
    args.share_embedder = True

    # configure for complete experiment and ablation models

    # get checkpoint save path
    args_for_checkpoint_folder_name = [
        args.net, args.data_type, args.copy_type,
        args.copy_loss_type, args.soft_copy_topN,
        args.only_copy_content,
        args.use_vocab_mask,
        args.use_clue_info, args.use_style_info,
        args.use_refine_copy_tgt, args.use_refine_copy_src, args.use_refine_copy_tgt_src,
        args.beam_size]  # NOTICE: change here. Keep the same with QG_main.py. Otherwise, there may be error.
    save_dir = args.checkpoint_dir
    args.checkpoint_dir = get_checkpoint_dir(save_dir, args_for_checkpoint_folder_name)
    # args.mode = "test"
    # if args.mode != "train":
    args.resume = args.checkpoint_dir + "model_best.pth.tar"  # !!!!! NOTICE: so set --resume won't change it.

    print(args)

    # set device, random seed, logger
    device, use_cuda, n_gpu = set_device(args.no_cuda)
    set_random_seed(args.seed)
    # logger = set_logger(args.log_file)
    logger = None

    # check whether need data preprocessing. If yes, preprocess data
    #if args.mode == "prepro":
    prepro(args, args.da_augmented_sentences_file, args.qg_augmented_sentences_file)
    #    return

    # data
    emb_mats = load(args.emb_mats_file)
    emb_dicts = load(args.emb_dicts_file)

    dataloader = get_loader(
        args, emb_dicts,
        args.qg_augmented_sentences_file, args.batch_size, shuffle=False)

    # model
    model = Model(args, emb_mats, emb_dicts)
    summarize_model(model)
    if use_cuda and args.use_multi_gpu and n_gpu > 1:
        model = nn.DataParallel(model)
    model.to(device)
    partial_models = None
    partial_resumes = None
    partial_trainables = None

    # optimizer and scheduler
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    for p in parameters:
        if p.dim() == 1:
            p.data.normal_(0, math.sqrt(6 / (1 + p.size(0))))
        elif list(p.shape) == [args.tgt_vocab_limit, 300]:
            print("omit embeddings.")
        else:
            nn.init.xavier_normal_(p, math.sqrt(3))
    optimizer = Optim(
        args.optim, args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        max_weight_value=args.max_weight_value,
        lr_decay=args.learning_rate_decay,
        start_decay_at=args.start_decay_at,
        decay_bad_count=args.halve_lr_bad_count
    )
    optimizer.set_parameters(model.parameters())
    scheduler = None

    loss = {}
    loss["P"] = torch.nn.CrossEntropyLoss()
    loss["D"] = torch.nn.BCEWithLogitsLoss(reduction="sum")

    # trainer
    trainer = Trainer(
        args,
        model,
        train_dataloader=None,
        dev_dataloader=None,
        loss=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        emb_dicts=emb_dicts,
        logger=logger,
        partial_models=partial_models,
        partial_resumes=partial_resumes,
        partial_trainables=partial_trainables)

    # start train/eval/test model
    start = datetime.now()
    args.use_ema = False
    trainer.test(dataloader, args.qg_result_file)
    get_qa_input_file(args.qg_result_file, args.da_paragraphs_file, args.qa_data_file)
    # TODO: delete duplicate examples. different clue, style may generate the same question...
    print(("Time of {} model: {}").format(args.mode, datetime.now() - start))


if __name__ == '__main__':
    main(parser.parse_args())

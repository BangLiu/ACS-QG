# Copyright (c) 2019-present
# Original codebase written by HuggingFace Inc. (https://github.com/huggingface/transfer-learning-conv-ai)
# Further reference: https://github.com/huggingface/transformers/blob/master/examples/run_lm_finetuning.py
# Forked and modified by Bang Liu

# python3 GPT2-QG/train.py --eval_before_start --n_epochs 4 --model_name_or_path gpt2 --output_dir ./gpt2_question_generation
# see: https://github.com/martiansideofthemoon/squash-generation/blob/master/schedulers/schedule_gpt2.sh

import os
import math
import logging
from pprint import pformat
from collections import defaultdict

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
import argparse
from .dataloader import get_dataset

from common.constants import EXP_PLATFORM
# !!! for running experiments on Venus
if EXP_PLATFORM.lower() == "venus":
    from pip._internal import main as pipmain
    pipmain(["install", "pytorch-ignite"])
    pipmain(["install", "transformers"])
from transformers import (WEIGHTS_NAME, CONFIG_NAME, AdamW, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler


#parser = argparse.ArgumentParser()
from .arguments import *

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
}


SPECIAL_TOKENS = ["<sos>", "<eos>", "<paragraph>", "<clue>", "<answer>", "<style>", "<question>", "<pad>"]
SPECIAL_TOKENS_DICT = {
    "bos_token": "<sos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "additional_special_tokens": ["<paragraph>", "<clue>", "<answer>", "<style>", "<question>"]
}
MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

logger = logging.getLogger(__file__)


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in MODEL_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -1] * (max_l - len(x)) for x in dataset[name]]
    return dataset


def build_input_from_segments(data_point, tokenizer, with_eos=True):
    """ Build a sequence of input.
        `<sos> .. paragraph text ..
         <clue> .. clue span ..
         <answer> .. answer span ..
         <style> .. question style ..
         <question> .. question span .. <eos>`
    """
    sos, eos, paragraph, clue, answer, style, question = \
        tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])

    curr_para = data_point['paragraph']
    curr_ans = data_point['answer']
    curr_ques = data_point['question']
    ans_start = data_point['answer_position_tokenized'][0]
    ans_end = data_point['answer_position_tokenized'][1]
    curr_style = data_point['style']

    clue_exist = (data_point['clue_start'] is not None)
    if clue_exist:
        curr_clue = data_point['clue']
        clue_start = data_point['clue_position_tokenized'][0]
        clue_end = data_point['clue_position_tokenized'][1]
    else:
        curr_clue = []

    # <sos> paragraph
    sequence = [sos] + curr_para
    # This segmentation will encode positional information
    token_types = [paragraph for i in range(len(curr_para) + 1)]
    if clue_exist:
        token_types[clue_start + 1:clue_end + 1] = [clue] * (clue_end - clue_start)
    token_types[ans_start + 1:ans_end + 1] = [answer] * (ans_end - ans_start)
    lm_labels = [-1 for _ in range(len(curr_para) + 1)]

    # <sos> paragraph <answer> answer
    sequence.extend([answer] + curr_ans)
    token_types.extend([answer for _ in range(len(curr_ans) + 1)])
    lm_labels.extend([-1 for _ in range(len(curr_ans) + 1)])

    # <sos> paragraph <answer> answer <clue> clue
    sequence.extend([clue] + curr_clue)
    token_types.extend([clue for _ in range(len(curr_clue) + 1)])
    lm_labels.extend([-1 for _ in range(len(curr_clue) + 1)])

    # <sos> paragraph <answer> answer <clue> clue <style> style
    sequence.extend([style] + curr_style)
    token_types.extend([style for _ in range(len(curr_style) + 1)])
    lm_labels.extend([-1 for _ in range(len(curr_style) + 1)])

    # <sos> paragraph <answer> answer <clue> clue <style> style <question> question [<eos>]
    if with_eos is True:
        sequence.extend([question] + curr_ques + [eos])
        token_types.extend([question for _ in range(len(curr_ques) + 2)])
        lm_labels.extend([-1] + curr_ques + [eos])
    else:
        sequence.extend([question] + curr_ques)
        token_types.extend([question for _ in range(len(curr_ques) + 1)])
        lm_labels.extend([-1] + curr_ques)

    assert len(sequence) == len(token_types)
    assert len(token_types) == len(lm_labels)

    instance = {
        "input_ids": sequence,
        "token_type_ids": token_types,
        "lm_labels": lm_labels
    }
    return instance, sequence


def build_para_only_input_from_segments(data_point, tokenizer):
    """A paragraph-only version of build_input_from_segments().
    `<sos> .. paragraph text ..`
    """
    sos, eos, paragraph, clue, answer, style, question = \
        tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])

    curr_para = data_point['paragraph']
    ans_start = data_point['answer_position_tokenized'][0]
    ans_end = data_point['answer_position_tokenized'][1]

    clue_exist = (data_point['clue_start'] is not None)
    if clue_exist:
        clue_start = data_point['clue_position_tokenized'][0]
        clue_end = data_point['clue_position_tokenized'][1]

    # <sos> paragraph
    sequence = [sos] + curr_para
    # This segmentation will encode positional information
    token_types = [paragraph for i in range(len(curr_para) + 1)]
    if clue_exist:
        token_types[clue_start + 1:clue_end + 1] = [clue] * (clue_end - clue_start)
    token_types[ans_start + 1:ans_end + 1] = [answer] * (ans_end - ans_start)
    lm_labels = [-1 for _ in range(len(curr_para) + 1)]

    assert len(sequence) == len(token_types)
    assert len(token_types) == len(lm_labels)

    instance = {
        "input_ids": sequence,
        "token_type_ids": token_types,
        "lm_labels": lm_labels
    }
    return instance, sequence


def build_acsq_only_input_from_segments(data_point, tokenizer, with_eos=True):
    """A answer-clue-style-question-only version of build_input_from_segments()."""
    sos, eos, paragraph, clue, answer, style, question = \
        tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])

    curr_ans = data_point['answer']
    curr_ques = data_point['question']
    clue_exist = (data_point['clue_start'] is not None)
    if clue_exist:
        curr_clue = data_point['clue']
    else:
        curr_clue = []
    curr_style = data_point['style']

    sequence = []
    token_types = []
    lm_labels = []

    # <sos> paragraph <answer> answer
    sequence.extend([answer] + curr_ans)
    token_types.extend([answer for _ in range(len(curr_ans) + 1)])
    lm_labels.extend([-1 for _ in range(len(curr_ans) + 1)])

    # <sos> paragraph <answer> answer <clue> clue
    sequence.extend([clue] + curr_clue)
    token_types.extend([clue for _ in range(len(curr_clue) + 1)])
    lm_labels.extend([-1 for _ in range(len(curr_clue) + 1)])

    # <sos> paragraph <answer> answer <clue> clue <style> style
    sequence.extend([style] + curr_style)
    token_types.extend([style for _ in range(len(curr_style) + 1)])
    lm_labels.extend([-1 for _ in range(len(curr_style) + 1)])

    # <sos> paragraph <answer> answer <clue> clue <style> style <question> question [<eos>]
    if with_eos is True:
        sequence.extend([question] + curr_ques + [eos])
        token_types.extend([question for _ in range(len(curr_ques) + 2)])
        lm_labels.extend([-1] + curr_ques + [eos])
    else:
        sequence.extend([question] + curr_ques)
        token_types.extend([question for _ in range(len(curr_ques) + 1)])
        lm_labels.extend([-1] + curr_ques)

    assert len(sequence) == len(token_types)
    assert len(token_types) == len(lm_labels)

    instance = {
        "input_ids": sequence,
        "token_type_ids": token_types,
        "lm_labels": lm_labels
    }
    return instance, sequence


def get_data_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    datasets_raw = {}
    logger.info("Loading training data")
    datasets_raw['train'] = get_dataset(tokenizer, args.train_dataset_cache, args.train_dataset_path, 'train', args.filetype, args.debug)
    logger.info("Loading validation data")
    datasets_raw['valid'] = get_dataset(tokenizer, args.dev_dataset_cache, args.dev_dataset_path, 'dev', args.filetype, args.debug)

    logger.info("Build inputs and labels")
    datasets = {
        "train": defaultdict(list),
        "valid": defaultdict(list)
    }

    for dataset_name, dataset in datasets_raw.items():
        for data_point in dataset:
            instance, _ = build_input_from_segments(data_point, tokenizer)
            for input_name, input_array in instance.items():
                datasets[dataset_name][input_name].append(input_array)

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler


def train():
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):  # !!!NOTICE: change output dir for each different settings.
        os.makedirs(args.output_dir)

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")

    # Load pretrained model and tokenizer
    config_class, model_class, tokenizer_class = MODEL_CLASSES["gpt2"]
    config = config_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    tokenizer.add_tokens(SPECIAL_TOKENS)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    # model.set_num_special_tokens(len(SPECIAL_TOKENS))
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    # optimizer = OpenAIAdam(model.parameters(), lr=args.lr)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer)

    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        cur_input_ids = batch[0]
        cur_lm_labels = batch[1]
        cur_token_type_ids = batch[2]
        model_outputs = model(input_ids=cur_input_ids, labels=cur_lm_labels, token_type_ids=cur_token_type_ids)
        lm_loss = model_outputs[0]
        loss = lm_loss / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, lm_labels, token_type_ids = batch

            # logger.info(tokenizer.decode(input_ids[0, :].tolist()))
            model_outputs = model(input_ids, token_type_ids=token_type_ids)
            lm_logits = model_outputs[0]

            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

            return lm_logits_flat_shifted, lm_labels_flat_shifted
    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {
        "nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1))
    }
    metrics.update({
        "average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)
    })
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        # tb_logger = TensorboardLogger(log_dir=args.output_dir)
        # tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        # tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        # tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(args.output_dir, 'checkpoint', save_interval=1, n_saved=3)  # !!!NOTICE: if fill exist, it will report error. set require_empty=False can avoid this.
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

        torch.save(args, args.output_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(args.output_dir, CONFIG_NAME))
        tokenizer.save_vocabulary(args.output_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(args.output_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        # tb_logger.close()


if __name__ == "__main__":
    train()

# -*- coding: utf-8 -*-
"""
Factorized question generation main file.
It trains a question generation model based on SQuAD or other datasets.
After training, it can generate questions for given dataset.
"""
import math
import torch
import torch.nn as nn
from datetime import datetime
from data_loader.FQG_data import prepro, get_loader
from trainer.FQG_trainer import Trainer
from optimizer.optim import Optim
from util.file_utils import load
from util.exp_utils import set_device, set_random_seed
from util.exp_utils import set_logger, summarize_model, get_checkpoint_dir
from config import *
from common.constants import EXP_PLATFORM


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
        args.beam_size]  # NOTICE: change here. Also notice, debug mode will replace the model.
    save_dir = args.checkpoint_dir
    args.checkpoint_dir = get_checkpoint_dir(save_dir, args_for_checkpoint_folder_name)
    if args.mode != "train":
        args.resume = args.checkpoint_dir + "model_best.pth.tar"  # !!!!! NOTICE: so set --resume won't change it.

    print(args)

    # set device, random seed, logger
    device, use_cuda, n_gpu = set_device(args.no_cuda)
    set_random_seed(args.seed)
    # logger = set_logger(args.log_file)
    logger = None

    # check whether need data preprocessing. If yes, preprocess data
    if args.not_processed_data:  # use --not_processed_data --spacy_not_processed_data for complete prepro
        prepro(args)

    # data
    emb_mats = load(args.emb_mats_file)
    emb_dicts = load(args.emb_dicts_file)

    train_dataloader = get_loader(
        args, emb_dicts,
        args.train_examples_file, args.batch_size, shuffle=True)
    dev_dataloader = get_loader(
        args, emb_dicts,
        args.dev_examples_file, args.batch_size, shuffle=False)
    test_dataloader = get_loader(
        args, emb_dicts,
        args.test_examples_file, args.batch_size, shuffle=False)

    # model
    model = Model(args, emb_mats, emb_dicts)
    summarize_model(model)
    if use_cuda and args.use_multi_gpu and n_gpu > 1:
        if EXP_PLATFORM.lower() == "venus":
            pass
        else:
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
        train_dataloader=train_dataloader,
        dev_dataloader=dev_dataloader,
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
    if args.mode == "train":
        trainer.train()
    elif args.mode == "eval_train":
        args.use_ema = False
        trainer.eval(train_dataloader, args.train_eval_file, args.train_output_file)
    elif args.mode in ["eval", "evaluation", "valid", "validation"]:
        args.use_ema = False
        trainer.eval(dev_dataloader, args.dev_eval_file, args.eval_output_file)
    elif args.mode == "test":
        args.use_ema = False
        trainer.eval(test_dataloader, args.test_eval_file, args.test_output_file)
    else:
        print("Error: set mode to be train or eval or test.")
    print(("Time of {} model: {}").format(args.mode, datetime.now() - start))


if __name__ == '__main__':
    main(parser.parse_args())

# -*- coding: utf-8 -*-
"""
Configuration of our project.
"""
import argparse
from common.constants import *


# data directory
# NOTICE: we assume a specific structure for project organization.
# Change the paths in this file and in common/constants.py if your
# project structure is different.
dataset_name = "SQuAD1.1-Zhou"  # NOTICE: change it for different datasets
original_data_folder = DATA_PATH + "original/" + dataset_name + "/"
processed_data_folder = DATA_PATH + "processed/" + dataset_name + "/"

# configure of embeddings
emb_config = {
    "word": {
        "emb_file": GLOVE_TXT_PATH,  # or None if we not use glove
        "emb_size": 20000,  # full size is int(2.2e6)
        "emb_dim": 300,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": False},
    "char": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 64,
        "trainable": True,
        "need_conv": True,
        "need_emb": True,
        "is_feature": False},
    "bpe": {
        "emb_file": BPE_EMB_PATH,
        "emb_size": 50509,
        "emb_dim": 100,
        "trainable": False,
        "need_conv": True,
        "need_emb": True,
        "is_feature": False},
    "pos": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": False},
    "ner": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": False},
    "iob": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 3,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": False},
    "dep": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": False},
    "is_lower": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": True},
    "is_stop": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": True},
    "is_punct": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": True},
    "is_digit": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": True},
    "like_num": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": True},
    "is_bracket": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": True},
    "is_overlap": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": True},
    "answer_iob": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": False},
    "is_clue": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": True},
    "is_clue_hard": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": True},
    "is_content": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": True}}

# embeddings in use
emb_tags = [
    "word", "answer_iob", "pos", "ner", "dep", "is_lower", "is_digit", "is_content"]

# embeddings that not countable during data pre-processing
emb_not_count_tags = {
    "is_overlap": [0.0, 1.0],
    "answer_iob": ["B", "I", "O"],
    "is_clue": [0.0, 1.0],
    "is_clue_hard": [0.0, 1.0],
    "is_clue_soft": [0.0, 1.0],
    "is_content": [0.0, 1.0]}

# parser used to read argument
parser = argparse.ArgumentParser(description='FactorizedQG')

# experiment
parser.add_argument(
    '--seed', type=int, default=12345)
parser.add_argument(
    '--mode',
    default='train', type=str,
    help='train, eval or test model (default: train)')
parser.add_argument(
    '--visualizer',
    default=False, action='store_true',
    help='use visdom visualizer or not')
parser.add_argument(
    '--log_file',
    default='log.txt',
    type=str, help='path of log file')
parser.add_argument(
    '--no_cuda',
    default=False, action='store_true',
    help='not use cuda')
parser.add_argument(
    '--use_multi_gpu',
    default=False, action='store_true',
    help='use multi-GPU in case there\'s multiple GPUs available')
parser.add_argument(
    '--debug',
    default=False, action='store_true',
    help='debug mode or not')
parser.add_argument(
    '--debug_batchnum',
    default=5, type=int,
    help='only train and test a few batches when debug (default: 5)')

# data
parser.add_argument(
    '--not_processed_data',
    default=False, action='store_true',
    help='whether the dataset already processed')
parser.add_argument(
    '--processed_by_spacy',
    default=False, action='store_true',
    help='whether the dataset already processed by spacy')
parser.add_argument(
    '--processed_example_features',
    default=False, action='store_true',
    help='whether the dataset examples are completely processed')
parser.add_argument(
    '--processed_emb',
    default=False, action='store_true',
    help='whether the embedding files already processed')
parser.add_argument(
    '--processed_related_words',
    default=False, action='store_true',
    help='whether the related words dict and ids mat already processed')

parser.add_argument(
    '--train_file',
    default=original_data_folder + 'train.txt',
    type=str, help='path of train dataset')
parser.add_argument(
    '--dev_file',
    default=original_data_folder + 'dev.txt',
    type=str, help='path of dev dataset')
parser.add_argument(
    '--test_file',
    default=original_data_folder + 'test.txt',
    type=str, help='path of test dataset')

parser.add_argument(
    '--train_examples_file',
    default=processed_data_folder + 'train-examples.pkl',
    type=str, help='path of train dataset examples file')
parser.add_argument(
    '--dev_examples_file',
    default=processed_data_folder + 'dev-examples.pkl',
    type=str, help='path of dev dataset examples file')
parser.add_argument(
    '--test_examples_file',
    default=processed_data_folder + 'test-examples.pkl',
    type=str, help='path of test dataset examples file')

parser.add_argument(
    '--train_spacy_processed_examples_file',
    default=processed_data_folder + 'train-spacy-processed-examples.pkl',
    type=str, help='path of train dataset spacy processed examples file')
parser.add_argument(
    '--dev_spacy_processed_examples_file',
    default=processed_data_folder + 'dev-spacy-processed-examples.pkl',
    type=str, help='path of dev dataset spacy processed examples file')
parser.add_argument(
    '--test_spacy_processed_examples_file',
    default=processed_data_folder + 'test-spacy-processed-examples.pkl',
    type=str, help='path of test dataset spacy processed examples file')

parser.add_argument(
    '--train_meta_file',
    default=processed_data_folder + 'train-meta.pkl',
    type=str, help='path of train dataset meta file')
parser.add_argument(
    '--dev_meta_file',
    default=processed_data_folder + 'dev-meta.pkl',
    type=str, help='path of dev dataset meta file')
parser.add_argument(
    '--test_meta_file',
    default=processed_data_folder + 'test-meta.pkl',
    type=str, help='path of test dataset meta file')

parser.add_argument(
    '--train_eval_file',
    default=processed_data_folder + 'train-eval.pkl',
    type=str, help='path of train dataset eval file')
parser.add_argument(
    '--dev_eval_file',
    default=processed_data_folder + 'dev-eval.pkl',
    type=str, help='path of dev dataset eval file')
parser.add_argument(
    '--test_eval_file',
    default=processed_data_folder + 'test-eval.pkl',
    type=str, help='path of test dataset eval file')

parser.add_argument(
    '--train_output_file',
    default=processed_data_folder + 'train_output.txt',
    type=str, help='path of train result file')
parser.add_argument(
    '--eval_output_file',
    default=processed_data_folder + 'eval_output.txt',
    type=str, help='path of evaluation result file')
parser.add_argument(
    '--test_output_file',
    default=processed_data_folder + 'test_output.txt',
    type=str, help='path of test result file')

parser.add_argument(
    '--emb_mats_file',
    default=processed_data_folder + 'emb_mats.pkl',
    type=str, help='path of embedding matrices file')
parser.add_argument(
    '--emb_dicts_file',
    default=processed_data_folder + 'emb_dicts.pkl',
    type=str, help='path of embedding dicts file')
parser.add_argument(
    '--counters_file',
    default=processed_data_folder + 'counters.pkl',
    type=str, help='path of counters file')

parser.add_argument(
    '--related_words_dict_file',
    default=processed_data_folder + 'related_words_dict.pkl',
    type=str, help='path of related_words_dict file')
parser.add_argument(
    '--related_words_ids_mat_file',
    default=processed_data_folder + 'related_words_ids_mat.pkl',
    type=str, help='path of related_words_ids_mat file')

parser.add_argument(
    '--lower',
    default=False, action='store_true',
    help='whether lowercase all texts in data')

# train
parser.add_argument(
    '-b', '--batch_size',
    default=32, type=int,
    help='mini-batch size (default: 32)')
parser.add_argument(
    '-e', '--epochs',
    default=10, type=int,
    help='number of total epochs (default: 20)')
parser.add_argument(
    '--val_num_examples',
    default=10000, type=int,
    help='number of examples for evaluation (default: 10000)')
parser.add_argument(
    '--save_freq',
    default=1, type=int,
    help='training checkpoint frequency (default: 1 epoch)')
parser.add_argument(
    '--checkpoint_dir',
    default=OUTPUT_PATH + 'checkpoint/', type=str,
    help='directory of saved model (default: checkpoint/)')
parser.add_argument(
    '--resume',
    default='', type=str,
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '--resume_partial',
    default=False, action='store_true',
    help='whether resume partial pretrained model component(s)')
parser.add_argument(
    '--print_freq',
    default=10, type=int,
    help='print training information frequency (default: 10 steps)')
parser.add_argument(
    '--lr',
    default=0.001, type=float,
    help='learning rate')
parser.add_argument(
    '--lr_warm_up_num',
    default=1000, type=int,
    help='number of warm-up steps of learning rate')
parser.add_argument(
    '--beta1',
    default=0.8, type=float,
    help='beta 1')
parser.add_argument(
    '--beta2',
    default=0.999, type=float,
    help='beta 2')
parser.add_argument(
    '--no_grad_clip',
    default=False, action='store_true',
    help='whether use gradient clip')
parser.add_argument(
    '--max_grad_norm',
    default=5.0, type=float,
    help='global Norm gradient clipping rate')
parser.add_argument(
    '--use_ema',
    default=False, action='store_true',
    help='whether use exponential moving average')
parser.add_argument(
    '--ema_decay',
    default=0.9999, type=float,
    help='exponential moving average decay')
parser.add_argument(
    '--use_early_stop',
    default=True, action='store_false',
    help='whether use early stop')
parser.add_argument(
    '--early_stop',
    default=20, type=int,
    help='checkpoints for early stop')

# learning rate
parser.add_argument(
    '-learning_rate', type=float, default=0.001,
    help="""Starting learning rate. If adagrad/adadelta/adam is
    used, then this is the global learning rate. Recommended
    settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""")
parser.add_argument(
    '-learning_rate_decay', type=float, default=0.5,
    help="""If update_learning_rate, decay learning rate by
    this much if (i) perplexity does not decrease on the
    validation set or (ii) epoch has gone past start_decay_at""")
parser.add_argument(
    '-start_decay_at', type=int, default=8,
    help="""Start decaying every epoch after and including this
    epoch""")
parser.add_argument(
    '-start_eval_batch', type=int, default=1000,
    help="""evaluate on dev per x batches.""")
parser.add_argument(
    '-eval_per_batch', type=int, default=500,
    help="""evaluate on dev per x batches.""")
parser.add_argument(
    '-halve_lr_bad_count', type=int, default=1,
    help="""for change lr.""")

# model
parser.add_argument(
    '--para_limit',
    default=400, type=int,
    help='maximum context token number')
parser.add_argument(
    '--ques_limit',
    default=50, type=int,
    help='maximum question token number')
parser.add_argument(
    '--ans_limit',
    default=30, type=int,
    help='maximum answer token number')
parser.add_argument(
    '--sent_limit',
    default=100, type=int,
    help='maximum sentence token number')
parser.add_argument(
    '--char_limit',
    default=16, type=int,
    help='maximum char number in a word')
parser.add_argument(
    '--bpe_limit',
    default=6, type=int,
    help='maximum bpe number in a word')
parser.add_argument(
    '--emb_config',
    default=emb_config, type=dict,
    help='config of embeddings')
parser.add_argument(
    '--emb_tags',
    default=emb_tags, type=list,
    help='tags of embeddings that we will use in model')
parser.add_argument(
    '--emb_not_count_tags',
    default=emb_not_count_tags, type=dict,
    help='tags of embeddings that we will not count by counter')
parser.add_argument(
    '--is_clue_topN',
    default=20, type=int,
    help='maximum glove distance for checking clue words')
parser.add_argument(
    '--soft_copy_topN',
    default=128, type=int,
    help='maximum glove distance for checking soft copy words')
parser.add_argument(
    '--max_topN',
    default=128, type=int,
    help='maximum glove distance for generating related words dict')

# tmp solution for load issue
parser.add_argument(
    '--d_model',
    default=128, type=int,
    help='model hidden size')
parser.add_argument(
    '--num_head',
    default=8, type=int,
    help='num head')
parser.add_argument(
    '-beam_size', type=int, default=5, help='Beam size')
parser.add_argument(
    '-layers', type=int, default=1,
    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument(
    '-enc_rnn_size', type=int, default=512,
    help='Size of LSTM hidden states')
parser.add_argument(
    '-dec_rnn_size', type=int, default=512,
    help='Size of LSTM hidden states')
parser.add_argument(
    '-att_vec_size', type=int, default=512,
    help='Concat attention vector sizes')
parser.add_argument(
    '-maxout_pool_size', type=int, default=2,
    help='Pooling size for MaxOut layer.')
parser.add_argument(
    '-input_feed', type=int, default=1,
    help="""Feed the context vector at each time step as
    additional input (via concatenation with the word
    embeddings) to the decoder.""")
parser.add_argument(
    '-brnn', action='store_true',
    help='Use a bidirectional encoder')
parser.add_argument(
    '-brnn_merge', default='concat',
    help="""Merge action for the bidirectional hidden states:
    [concat|sum]""")
parser.add_argument(
    '-optim', default='adam',
    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument(
    '-max_weight_value', type=float, default=15,
    help="""If the norm of the gradient vector exceeds this,
    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument(
    '-dropout', type=float, default=0.5,
    help='Dropout probability; applied between LSTM stacks.')
parser.add_argument(
    '-curriculum', type=int, default=0,
    help="""For this many epochs, order the minibatches based
    on source sequence length. Sometimes setting this to 1 will
    increase convergence speed.""")
parser.add_argument(
    '-extra_shuffle', action="store_true",
    help="""By default only shuffle mini-batch order; when true,
    shuffle and re-assign mini-batches""")

# tricks
parser.add_argument(
    '--use_clue_info',
    default=False, action='store_true',
    help='whether use clue information')
parser.add_argument(
    '--use_style_info',
    default=False, action='store_true',
    help='whether use style information')
parser.add_argument(
    '--use_content_separator',
    default=False, action='store_true',
    help='whether use content separator')
parser.add_argument(
    '--use_soft_copy',
    default=False, action='store_true',
    help='whether use soft copy mechanism')
parser.add_argument(
    '--copy_type',
    default='hard-oov', type=str,
    help='which copy mechanism to use: hard-oov, hard, soft-oov, soft')
parser.add_argument(
    '--copy_loss_type',
    default=1, type=int,
    help='the type of copy loss function')
parser.add_argument(
    '--only_copy_content',
    default=False, action='store_true',
    help='whether only copy content words in copy mechanism')
parser.add_argument(
    '--use_vocab_mask',
    default=False, action='store_true',
    help='whether mask input content words in vocab')

parser.add_argument(
    '--tgt_vocab_limit',
    default=20000, type=int,
    help='maximum vocab size of target words in seq2seq')
parser.add_argument(
    '--share_embedder',
    default=False, action='store_true',
    help='encoder decoder share embedder')

parser.add_argument(
    '--use_answer_separate',
    default=False, action='store_true',
    help='whether use answer separate trick')

parser.add_argument(
    '--num_question_style',
    default=9, type=int,
    help='number of different question types')
parser.add_argument(
    '--style_emb_dim',
    default=16, type=int,
    help='embedding dimension for question styles')

parser.add_argument(
    '--clue_coef',
    default=1.0, type=float, help='clue loss coef')
parser.add_argument(
    '--style_coef',
    default=1.0, type=float, help='style loss coef')
parser.add_argument(
    '--data_type',
    default='squad', type=str,
    help='which dataset to use')
parser.add_argument(
    '--net',
    default='FQG', type=str,
    help='which neural network model to use')

parser.add_argument(
    '--use_refine_copy',
    default=False, action='store_true',
    help='whether refine copy switch and tgt')
parser.add_argument(
    '--use_refine_copy_tgt',
    default=False, action='store_true',
    help='whether refine copy and tgt vocab')
parser.add_argument(
    '--use_refine_copy_src',
    default=False, action='store_true',
    help='whether refine copy and src vocab')
parser.add_argument(
    '--use_refine_copy_tgt_src',
    default=False, action='store_true',
    help='whether refine copy, tgt, and src vocab')
parser.add_argument(
    '--refined_copy_vocab_limit',
    default=2000, type=int,
    help='refined maximum vocab size of copy tgt words in seq2seq')
parser.add_argument(
    '--refined_tgt_vocab_limit',
    default=2000, type=int,
    help='refined maximum vocab size of target words in seq2seq')
parser.add_argument(
    '--refined_src_vocab_limit',
    default=2000, type=int,
    help='refined maximum vocab size of src words in seq2seq')

# doing experiments for paper
parser.add_argument(
    '--experiment',
    default=False, action='store_true', help='do experiment for complete model')
parser.add_argument(
    '--ablation_no_clue',
    default=False, action='store_true', help='remove clue information')
parser.add_argument(
    '--ablation_no_answer',
    default=False, action='store_true', help='remove answer information')
parser.add_argument(
    '--ablation_no_style',
    default=False, action='store_true', help='remove style information')
parser.add_argument(
    '--ablation_no_cfseparate',
    default=False, action='store_true', help='remove content function separation information')
parser.add_argument(
    '--ablation_no_softcopy',
    default=False, action='store_true', help='remove soft copy mechanism')

# Data Augmentation parameters
parser.add_argument(
    '--da_max_num',
    default=100000, type=int,
    help='maxinum number of augmented examples')
parser.add_argument(
    '--da_input_file',
    default='', type=str,
    help='path to input file')
parser.add_argument(
    '--da_input_type',
    default='', type=str,
    help='type of input file: wiki10000 or squad')
parser.add_argument(
    '--da_sentences_file',
    default='', type=str,
    help='path to output file')
parser.add_argument(
    '--da_paragraphs_file',
    default='', type=str,
    help='path to output file')
parser.add_argument(
    '--da_augmented_sentences_file',
    default='', type=str,
    help='path to output file')
parser.add_argument(
    '--da_task',
    default='', type=str,
    help='file2sentences or sentences2augmented_sentences')
parser.add_argument(
    '--da_start_index',
    default=0, type=int,
    help='start index to transform sentences into augmented sentences.')
parser.add_argument(
    '--da_end_index',
    default=10000, type=int,
    help='end index to transform sentences into augmented sentences.')
parser.add_argument(
    '--num_sample_answer',
    default=5, type=int,
    help='maximum number of augmented answers for each sentence')
parser.add_argument(
    '--num_sample_clue',
    default=2, type=int,
    help='maximum number of augmented clues for each augmented answer')
parser.add_argument(
    '--num_sample_style',
    default=2, type=int,
    help='maximum number of augmented question type for each augmented clue')
parser.add_argument(
    '--max_sample_times',
    default=20, type=int,
    help='maximum number of random samples to sample enough different info')
parser.add_argument(
    '--not_processed_sample_probs_file',
    default=False, action='store_true', help='whether processed sample probs file')

# QG_augment parameters
parser.add_argument(
    '--qg_augmented_sentences_file',
    default='', type=str,
    help='file name of the processed augmented sentences pkl')
parser.add_argument(
    '--qg_result_file',
    default='', type=str,
    help='file name of the question generation result txt file')
parser.add_argument(
    '--qa_data_file',
    default='', type=str,
    help='file name of the question answering input data txt file')

# -*- coding: utf-8 -*-
"""
Data augmenter main file.
It handles SQuAD or other datasets to select all possible answers, clues, and question types.
The selected data can be read by QG data loader, and generate questions.
"""
# !!! for running experiments on Venus
from config import *
from common.constants import EXP_PLATFORM
if EXP_PLATFORM.lower() == "venus":
    from nltk import data
    data.path.append('./nltk_need/nltk_data/')

import json
import nltk
import codecs
from tqdm import tqdm
from data_augmentor.FQG_data_augmentor import augment_qg_data, get_sample_probs
from util.file_utils import save, load


def normalize_text(text):
    """
    Replace some special characters in text.
    """
    # NOTICE: don't change the text length.
    # Otherwise, the answer position is changed.
    text = text.replace("''", '" ').replace("``", '" ').replace("\t", " ")
    return text


def wiki2sentences(input_path, output_path, paragraphs_path, max_length=100, min_length=5, max_plength=400, min_plength=5):
    outfile = open(output_path, 'w', encoding='utf8')
    outfile_p = open(paragraphs_path, 'w', encoding='utf8')
    with codecs.open(input_path, encoding='utf8') as infile:
        data = json.load(infile)
    pid = 0
    sid = 0
    for k in data:
        paragraph_list = data[k]
        for p in paragraph_list:
            len_p = len(p.split())
            if len_p >= max_plength or len_p <= min_plength:
                continue
            p = normalize_text(p)
            outfile_p.write(str(pid) + "\t" + p.rstrip().replace("\n", "\\n") + "\n")
            sentences = nltk.sent_tokenize(p)
            for s in sentences:
                len_s = len(s.split())
                if len_s >= max_length or len_s <= min_length:
                    continue
                s = normalize_text(s)
                outfile.write(str(pid) + "\t" + str(sid) + "\t" + s.rstrip().replace("\n", "\\n") + "\n")
                sid += 1
            pid += 1
    infile.close()
    outfile.close()
    outfile_p.close()


def squad2sentences(input_path, output_path, paragraphs_path,
                    max_length=100, min_length=5, max_plength=400, min_plength=5):
    outfile = open(output_path, 'w', encoding='utf8')
    outfile_p = open(paragraphs_path, 'w', encoding='utf8')
    with codecs.open(input_path, "r", encoding='utf8') as infile:
        source = json.load(infile)
        pid = 0
        sid = 0
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"]
                p = context
                len_p = len(p.split())
                if len_p >= max_plength or len_p <= min_plength:
                    continue
                p = normalize_text(p)
                outfile_p.write(str(pid) + "\t" + p.rstrip().replace("\n", "\\n") + "\n")
                sentences = nltk.sent_tokenize(context)
                for s in sentences:
                    len_s = len(s.split())
                    if len_s >= max_length or len_s <= min_length:
                        continue
                    s = normalize_text(s)
                    outfile.write(str(pid) + "\t" + str(sid) + "\t" + s.rstrip().replace("\n", "\\n") + "\n")
                    sid += 1
                pid += 1
    infile.close()
    outfile.close()
    outfile_p.close()


def file2sentences(input_path, data_type, output_path, paragraphs_path,
                   max_length=100, min_length=5, max_plength=400, min_plength=5):
    if data_type.lower() == "wiki10000":
        wiki2sentences(input_path, output_path, paragraphs_path, max_length, min_length, max_plength, min_plength)
    elif data_type.lower() == "squad":
        squad2sentences(input_path, output_path, paragraphs_path, max_length, min_length, max_plength, min_plength)
    else:
        print("The data_type must be wiki10000 or squad...")


def sentences2augmented_sentences(input_path, output_path, start_index, end_index, sample_probs,
                                  num_sample_answer=5, num_sample_clue=2, num_sample_style=2,
                                  max_sample_times=20):
    augmented_sentences = []
    with codecs.open(input_path, "r", encoding='utf8') as infile:
        sentences = infile.readlines()
        assert start_index < end_index
        assert start_index < len(sentences)
        assert end_index <= len(sentences)
        print("Start augment data...")
        for i in range(start_index, end_index):
            print(i)
            s_split = sentences[i].rstrip().split("\t")
            pid = s_split[0]
            sid = s_split[1]
            s = s_split[2]
            # augmented_s = augment_qg_data(s)  # NOTICE: for FQG_data_augmentor_old
            augmented_s = augment_qg_data(
                s, sample_probs,
                num_sample_answer, num_sample_clue, num_sample_style,
                max_sample_times)
            augmented_s["pid"] = pid
            augmented_s["sid"] = sid
            augmented_sentences.append(augmented_s)
    save(output_path, augmented_sentences, "save augmented sentences...")
    infile.close()


def main(args):
    # prepro files
    CURRENT_PATH = os.getcwd().split("/")
    DATA_PATH = "/".join(CURRENT_PATH[:-4]) + "/Datasets/"

    DATA_ACS_INFO_FILE_PATH = DATA_PATH + "processed/SQuAD1.1-Zhou/squad_ans_clue_style_info.pkl"
    SAMPLE_PROBS_FILE_PATH = DATA_PATH + "processed/SQuAD1.1-Zhou/squad_sample_probs.pkl"
    SQUAD_FILE = DATA_PATH + "original/SQuAD1.1-Zhou/train.txt"

    # !!!NOTICE: remember to clear these files when needed, otherwise we won't re-calculate.
    if not os.path.isfile(SAMPLE_PROBS_FILE_PATH) or args.not_processed_sample_probs_file:
        print(SAMPLE_PROBS_FILE_PATH + " not exist.\nNow start generate these files.\n")
        # if not exist, generate mapping dict and save to file
        get_sample_probs(
            filename=SQUAD_FILE, filetype="squad",
            save_dataset_info_file=DATA_ACS_INFO_FILE_PATH, save_sample_probs_file=SAMPLE_PROBS_FILE_PATH,
            sent_limit=100, ques_limit=50, answer_limit=30, is_clue_topN=20,
            debug=args.debug, debug_length=20,
            answer_length_bin_width=3, answer_length_min_val=0, answer_length_max_val=30,
            clue_dep_dist_bin_width=2, clue_dep_dist_min_val=0, clue_dep_dist_max_val=20)

    SAMPLE_PROBS = load(SAMPLE_PROBS_FILE_PATH)
    print(SAMPLE_PROBS_FILE_PATH + " loaded.\n")

    # excute tasks
    if args.debug:
        args.da_start_index = 0
        args.da_end_index = 10

    if args.da_task == "file2sentences":
        file2sentences(
            args.da_input_file,
            args.da_input_type,
            args.da_sentences_file,
            args.da_paragraphs_file,
            max_plength=args.para_limit,
            max_length=args.sent_limit)
    if args.da_task == "sentences2augmented_sentences":
        sentences2augmented_sentences(
            args.da_sentences_file,
            args.da_augmented_sentences_file,
            args.da_start_index,
            args.da_end_index,
            SAMPLE_PROBS,
            args.num_sample_answer,
            args.num_sample_clue,
            args.num_sample_style,
            args.max_sample_times)


if __name__ == '__main__':
    main(parser.parse_args())

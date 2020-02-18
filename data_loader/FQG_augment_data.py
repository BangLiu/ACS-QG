# -*- coding: utf-8 -*-
"""
Load augmented datasets for generating questions with trained QG model.
"""
import random
import torch
import copy
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from .config import *
from util.file_utils import load, save
from util.prepro_utils import *
from .FQG_data_utils import *
from common.constants import NLP, FUNCTION_WORDS_LIST, OUTPUT_PATH, Q_TYPE2ID_DICT
from .FQG_data import write_example


def get_spacy_processed_examples(config, augmented_sentences,
                                 debug=False, debug_length=20, shuffle=False):
    print("Start transform augmented sentences to spaCy processed examples...")
    start = datetime.now()
    examples = []
    i = 0
    for s in tqdm(augmented_sentences):
        if "ans_sent_doc" not in s:
            s["ans_sent_doc"] = NLP(s["context"])
        s["ans_sent_tokens"] = [token.text for token in s["ans_sent_doc"]]
        examples.append(s)
        i = i + 1
        if debug and i >= debug_length:
            break
    if shuffle:
        random.shuffle(augmented_sentences)

    print(("Time of get spaCy processed examples: {}").format(
        datetime.now() - start))
    print("Number of spaCy processed examples: ", len(examples))
    return examples


def build_linguistic_features(config, example, emb_dicts):
    """
    Given an example, we get its features / tags, and ids.
    """
    # feature settings
    fields = ["ans_sent"]
    length_limits = {
        "ques": config.ques_limit,
        "answer": config.ans_limit,
        "ans_sent": config.sent_limit,
        "word": config.char_limit,
        "bpe": config.bpe_limit}
    tags = config.emb_config.keys()

    for field in fields:
        # start, end = example["y1_in_sent"], example["y2_in_sent"]
        for tag in tags:
            field_id = field + "_" + tag + "_ids"
            field_tag = field + "_" + tag
            field_length = len(example[field + "_tokens"])  # unpadded length
            if tag == "word":
                example[field_id] = spacydoc2wids(
                    example[field + "_doc"], emb_dicts[tag],
                    length_limits[field])
            elif tag == "char":
                example[field_id] = spacydoc2cids(
                    example[field + "_doc"], emb_dicts[tag],
                    length_limits[field], length_limits["word"])
            elif tag == "bpe":
                example[field_id] = spacydoc2bpeids(
                    example[field + "_doc"], emb_dicts[tag],
                    length_limits[field], length_limits["bpe"])
            # elif tag == "answer_iob":
            #     if field == "ans_sent":
            #         example[field_tag] = get_answer_iob(
            #             field_length, start, end)
            #         example[field_id] = feature2ids(
            #             example[field_tag], emb_dicts[tag],
            #             field_length, length_limits[field])
            elif tag in ["pos", "ner", "iob", "dep"]:
                example[field_id] = spacydoc2tagids(
                    example[field + "_doc"], tag, emb_dicts[tag],
                    length_limits[field])
            elif tag in ["is_alpha", "is_ascii", "is_digit", "is_lower",
                         "is_title", "is_punct", "is_left_punct",
                         "is_right_punct", "is_bracket", "is_quote",
                         "is_currency", "is_stop", "like_url", "like_num",
                         "like_email"]:
                example[field_tag] = spacydoc2features(
                    example[field + "_doc"], tag, length_limits[field])
                example[field_id] = feature2ids(
                    example[field_tag], emb_dicts[tag],
                    field_length, length_limits[field])
            else:
                pass
    # NOTICE: here we use lower
    example["src_tokens"] = [x.lower() for x in example["ans_sent_tokens"]]
    return example


def build_fqg_features(config, example, emb_dicts):
    """
    Get ans_sent is_content, is_clue
    """
    example["ans_sent_is_content"] = get_content_ids(
        example["ans_sent_doc"], FUNCTION_WORDS_LIST, config.sent_limit)

    # get ids for embedding layer
    example["ans_sent_is_content_ids"] = feature2ids(
        example["ans_sent_is_content"], emb_dicts["is_content"],
        len(example["ans_sent_doc"]), config.sent_limit)

    return example


def get_featured_examples(config, examples, emb_dicts):
    total = 0
    total_ = 0
    examples_with_features = []

    for example in tqdm(examples):
        total_ += 1
        example = build_linguistic_features(config, example, emb_dicts)
        example = build_fqg_features(config, example, emb_dicts)

        # NOTICE!!!! maybe too much combinations
        example["selected_info_processed"] = []

        for info in example["selected_infos"]:
            answer_text = info["answer"]["answer_text"]
            char_start = info["answer"]["char_start"]
            char_end = info["answer"]["char_end"]
            answer_bio_ids = info["answer"]["answer_bio_ids"]
            answer_chunk_tag = info["answer"]["answer_chunk_tag"]

            # filter
            answer_length = answer_bio_ids.count("B") + answer_bio_ids.count("I")
            if (len(example["ans_sent_doc"]) > config.sent_limit or answer_length > config.ans_limit):
                continue
            total += 1

            for clue in info["clues"]:
                clue_text = clue["clue_text"]
                clue_binary_id = clue["clue_binary_ids"]
                clue_binary_id_padded = np.zeros([config.sent_limit], dtype=np.float32)
                clue_binary_id_padded[:min(len(clue_binary_id), config.sent_limit)] = clue_binary_id[:config.sent_limit]

                for style_text in info["styles"]:
                    style_id = Q_TYPE2ID_DICT[style_text]

                    processed_info = {}
                    processed_info["ans_sent_is_clue"] = clue_binary_id_padded
                    processed_info["clue_text"] = clue_text
                    processed_info["ans_sent_answer_iob"] = answer_bio_ids
                    processed_info["answer_text"] = answer_text
                    processed_info["char_start"] = char_start
                    processed_info["char_end"] = char_end
                    processed_info["answer_chunk_tag"] = answer_chunk_tag
                    processed_info["ques_type"] = style_text
                    processed_info["ques_type_id"] = style_id

                    processed_info["ans_sent_is_clue_ids"] = feature2ids(
                        processed_info["ans_sent_is_clue"], emb_dicts["is_clue"],
                        len(example["ans_sent_doc"]), config.sent_limit)
                    processed_info["ans_sent_answer_iob_ids"] = feature2ids(
                        processed_info["ans_sent_answer_iob"], emb_dicts["answer_iob"],
                        len(example["ans_sent_doc"]), config.sent_limit)
                    example["selected_info_processed"].append(processed_info)

        example["selected_infos"] = None
        examples_with_features.append(example)
        print("len examples_with_features:  ", len(examples_with_features))

    print("Built {} / {} instances of features in total".format(total, total_))
    return examples_with_features


def prepro(config, augmented_sentences_pkl_file, processed_augmented_sentences_pkl_file):
    debug = config.debug
    debug_length = config.debug_batchnum * config.batch_size

    # get train spacy processed examples and counters
    examples = load(augmented_sentences_pkl_file)
    examples = get_spacy_processed_examples(config, examples, debug, debug_length, shuffle=False)

    # get emb_mats and emb_dicts
    emb_dicts = load(config.emb_dicts_file)

    # get featured examples
    examples = get_featured_examples(config, examples, emb_dicts)
    save(processed_augmented_sentences_pkl_file, examples, message="processed_augmented_sentences_pkl_file")

    # print to txt to debug
    # write_example(examples[5], OUTPUT_PATH + "debug/augment_example.txt")


class QGData_augment(Dataset):

    def __init__(self, config, emb_dicts, examples_file):
        self.examples = load(examples_file)
        self.num_auginfos_per_example = [len(e["selected_info_processed"]) for e in self.examples]
        self.num_auginfos = np.cumsum(self.num_auginfos_per_example)
        self.num = self.num_auginfos[-1]
        # print(self.examples[0])

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        e_idx = (self.num_auginfos > idx).tolist().index(True)
        info_idx = self.num_auginfos[e_idx] - idx - 1
        return (self.examples[e_idx]["ans_sent_word_ids"],
                self.examples[e_idx]["ans_sent_char_ids"],
                self.examples[e_idx]["ans_sent_bpe_ids"],
                self.examples[e_idx]["ans_sent_pos_ids"],
                self.examples[e_idx]["ans_sent_ner_ids"],
                self.examples[e_idx]["ans_sent_iob_ids"],
                self.examples[e_idx]["ans_sent_dep_ids"],
                self.examples[e_idx]["selected_info_processed"][info_idx]["ans_sent_answer_iob_ids"],
                self.examples[e_idx]["ans_sent_is_digit"],
                self.examples[e_idx]["ans_sent_is_digit_ids"],
                self.examples[e_idx]["ans_sent_is_lower"],
                self.examples[e_idx]["ans_sent_is_lower_ids"],
                self.examples[e_idx]["ans_sent_is_punct"],
                self.examples[e_idx]["ans_sent_is_punct_ids"],
                self.examples[e_idx]["ans_sent_is_bracket"],
                self.examples[e_idx]["ans_sent_is_bracket_ids"],
                self.examples[e_idx]["ans_sent_is_stop"],
                self.examples[e_idx]["ans_sent_is_stop_ids"],
                self.examples[e_idx]["ans_sent_like_num"],
                self.examples[e_idx]["ans_sent_like_num_ids"],
                # self.examples[idx]["ans_sent_syntactic_edges"],
                self.examples[e_idx]["ans_sent_is_content"],
                self.examples[e_idx]["ans_sent_is_content_ids"],
                # self.examples[idx]["ans_sent_is_clue_hard"],
                # self.examples[idx]["ans_sent_is_clue_hard_ids"],
                self.examples[e_idx]["selected_info_processed"][info_idx]["ans_sent_is_clue"],
                self.examples[e_idx]["selected_info_processed"][info_idx]["ans_sent_is_clue_ids"],

                self.examples[e_idx]["selected_info_processed"][info_idx]["ques_type_id"],
                self.examples[e_idx]["sid"],
                self.examples[e_idx]["pid"],
                self.examples[e_idx]["src_tokens"],
                self.examples[e_idx]["context"],
                self.examples[e_idx]["selected_info_processed"][info_idx]["answer_text"],
                self.examples[e_idx]["selected_info_processed"][info_idx]["char_start"],
                self.examples[e_idx]["selected_info_processed"][info_idx]["char_end"])


def collate(data):
    (ans_sent_word_ids, ans_sent_char_ids, ans_sent_bpe_ids,
     ans_sent_pos_ids, ans_sent_ner_ids, ans_sent_iob_ids,
     ans_sent_dep_ids, ans_sent_answer_iob_ids,
     ans_sent_is_digit, ans_sent_is_digit_ids,
     ans_sent_is_lower, ans_sent_is_lower_ids,
     ans_sent_is_punct, ans_sent_is_punct_ids,
     ans_sent_is_bracket, ans_sent_is_bracket_ids,
     ans_sent_is_stop, ans_sent_is_stop_ids,
     ans_sent_like_num, ans_sent_like_num_ids,
     # ans_sent_syntactic_edges,
     ans_sent_is_content, ans_sent_is_content_ids,
     # ans_sent_is_clue_hard, ans_sent_is_clue_hard_ids,
     ans_sent_is_clue, ans_sent_is_clue_ids,

     ques_type_id,
     sid, pid, src_tokens,
     ans_sent, answer_text, char_start, char_end) = zip(*data)
    batch = {}
    batch["ans_sent_word_ids"] = torch.LongTensor(ans_sent_word_ids)
    batch["ans_sent_char_ids"] = torch.LongTensor(ans_sent_char_ids)
    batch["ans_sent_bpe_ids"] = torch.LongTensor(ans_sent_bpe_ids)
    batch["ans_sent_pos_ids"] = torch.LongTensor(ans_sent_pos_ids)
    batch["ans_sent_ner_ids"] = torch.LongTensor(ans_sent_ner_ids)
    batch["ans_sent_iob_ids"] = torch.LongTensor(ans_sent_iob_ids)
    batch["ans_sent_dep_ids"] = torch.LongTensor(ans_sent_dep_ids)
    batch["ans_sent_answer_iob_ids"] = torch.LongTensor(ans_sent_answer_iob_ids)
    batch["ans_sent_is_digit"] = torch.FloatTensor(ans_sent_is_digit)
    batch["ans_sent_is_digit_ids"] = torch.LongTensor(ans_sent_is_digit_ids)
    batch["ans_sent_is_lower"] = torch.FloatTensor(ans_sent_is_lower)
    batch["ans_sent_is_lower_ids"] = torch.LongTensor(ans_sent_is_lower_ids)
    batch["ans_sent_is_punct"] = torch.FloatTensor(ans_sent_is_punct)
    batch["ans_sent_is_punct_ids"] = torch.LongTensor(ans_sent_is_punct_ids)
    batch["ans_sent_is_bracket"] = torch.FloatTensor(ans_sent_is_bracket)
    batch["ans_sent_is_bracket_ids"] = torch.LongTensor(ans_sent_is_bracket_ids)
    batch["ans_sent_is_stop"] = torch.FloatTensor(ans_sent_is_stop)
    batch["ans_sent_is_stop_ids"] = torch.LongTensor(ans_sent_is_stop_ids)
    batch["ans_sent_like_num"] = torch.FloatTensor(ans_sent_like_num)
    batch["ans_sent_like_num_ids"] = torch.LongTensor(ans_sent_like_num_ids)
    # batch["ans_sent_syntactic_edges"] = ans_sent_syntactic_edges
    batch["ans_sent_is_content"] = torch.LongTensor(ans_sent_is_content)
    batch["ans_sent_is_content_ids"] = torch.LongTensor(ans_sent_is_content_ids)
    # batch["ans_sent_is_clue_hard"] = ans_sent_is_clue_hard
    # batch["ans_sent_is_clue_hard_ids"] = ans_sent_is_clue_hard_ids
    batch["ans_sent_is_clue"] = torch.FloatTensor(ans_sent_is_clue)
    batch["ans_sent_is_clue_ids"] = torch.LongTensor(ans_sent_is_clue_ids)

    batch["ques_type_id"] = torch.LongTensor(ques_type_id)
    batch["sid"] = sid
    batch["pid"] = pid
    batch["src_tokens"] = src_tokens

    batch["ans_sent"] = ans_sent
    batch["answer_text"] = answer_text
    batch["char_start"] = char_start
    batch["char_end"] = char_end
    return batch


def get_loader(config, emb_dicts, examples_file, batch_size, shuffle=False):
    dataset = QGData_augment(config, emb_dicts, examples_file)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate)
    return data_loader

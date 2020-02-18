# -*- coding: utf-8 -*-
"""
Question evaluator main file.
It calculates the quality metrics for each generated (context, question, answer) sample, and filters low quality samples.
"""
from common.constants import EXP_PLATFORM
# !!! for running experiments on Venus
if EXP_PLATFORM.lower() == "venus":
    from pip._internal import main as pipmain
    pipmain(["install", "textstat"])

import argparse
# import textacy
import textstat
import math
import torch
import codecs
from datetime import datetime
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
from data_augmentor.style_selector import get_answer_ner_tag
from data_loader.FQG_data_utils import get_question_type
from util.file_utils import load


parser = argparse.ArgumentParser()

parser.add_argument("--input_file", default=None, type=str, required=True,
                    help="The input data file.")
parser.add_argument("--input_augmented_pkl_file", default=None, type=str, required=True,
                    help="The input augmented data pkl file")
parser.add_argument("--output_file", default=None, type=str, required=True,
                    help="The output file ")

# Load pre-trained model (weights)
if EXP_PLATFORM.lower() == "venus":
    model = GPT2LMHeadModel.from_pretrained('../../../models/gpt2/')
else:
    model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

# Load pre-trained model tokenizer (vocabulary)
if EXP_PLATFORM.lower() == "venus":
    tokenizer = GPT2Tokenizer.from_pretrained('../../../models/gpt2/')
else:
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


def get_readibility(text, metric="flesch_kincaid_grade"):
    """
    Return a score which reveals a piece of text's readability level.
    Reference: https://chartbeat-labs.github.io/textacy/getting_started/quickstart.html
               https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
    """
    if metric == "flesch_kincaid_grade":
        result = textstat.flesch_kincaid_grade(text)
    elif metric == "flesch_reading_ease":
        result = textstat.flesch_reading_ease(text)
    elif metric == "smog_index":
        result = textstat.smog_index(text)
    elif metric == "coleman_liau_index":
        result = textstat.coleman_liau_index(text)
    elif metric == "automated_readability_index":
        result = textstat.automated_readability_index(text)
    elif metric == "dale_chall_readability_score":
        result = textstat.dale_chall_readability_score(text)
    elif metric == "difficult_words":
        result = textstat.difficult_words(text)
    elif metric == "linsear_write_formula":
        result = textstat.linsear_write_formula(text)
    elif metric == "gunning_fog":
        result = textstat.gunning_fog(text)
    elif metric == "text_standard":
        result = textstat.text_standard(text)
    else:
        print("ERROR: Please select correct metric!")
        result = None
    return result


def get_perplexity(sentence):
    """
    NOTICE: for inputs like "," will have error. Don't know how to solve yet. Maybe bug of huggingface or GPT2.
    Currently, I just use try except to avoid the problem.
    """
    try:
        tokenize_input = tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        loss = model(tensor_input, labels=tensor_input)
        return math.exp(loss[0].item() / len(tokenize_input))
    except:
        return 1000000000


def evaluate_and_filter(input_file, input_augmented_pkl_file, output_file):
    augmented_examples = load(input_augmented_pkl_file)
    pid_sid_ans2ans_labels = {}
    for e in augmented_examples:
        pid = e["pid"]
        sid = e["sid"]
        for info in e["selected_info_processed"]:
            ans = info["answer_text"]
            compound_id = str(pid) + "_" + str(sid) + "_" + str(ans)
            ans_chunk_label = info["answer_chunk_tag"]
            ans_ner_label = get_answer_ner_tag(e["ans_sent_doc"], ans, processed_by_spacy=True)
            if compound_id in pid_sid_ans2ans_labels:
                pass
            else:
                pid_sid_ans2ans_labels[compound_id] = {"ans_chunk_label": ans_chunk_label, "ans_ner_label": ans_ner_label}

    outfile = open(output_file, 'w', encoding='utf8')
    with codecs.open(input_file, encoding='utf8') as infile:
        lines = infile.readlines()
        i = 0
        for line in lines:
            line_split = str(line).rstrip().split("\t")
            example_pid = line_split[0]  # same with get_qa_input_file in QG_augment_main.py
            example_sid = line_split[1]
            q = line_split[2]
            example_ans_sent = line_split[3]
            example_answer_text = line_split[4]
            example_paragraph = line_split[7]

            paragraph_readibility = get_readibility(example_paragraph)
            paragraph_perplexity = get_perplexity(example_paragraph)
            paragraph_length = len(example_paragraph.split())
            ans_sent_readibility = get_readibility(example_ans_sent)
            ans_sent_perplexity = get_perplexity(example_ans_sent)
            ans_sent_length = len(example_ans_sent.split())

            question_readibility = get_readibility(q)
            question_perplexity = get_perplexity(q)
            question_length = len(q.split())
            question_type_text, question_type_id = get_question_type(q)

            answer_readibility = get_readibility(example_answer_text)
            answer_perplexity = get_perplexity(example_answer_text)
            compound_id = str(example_pid) + "_" + str(example_sid) + "_" + str(example_answer_text)
            answer_chunk_tag = pid_sid_ans2ans_labels[compound_id]["ans_chunk_label"]
            answer_ner_tag = pid_sid_ans2ans_labels[compound_id]["ans_ner_label"]
            answer_length = len(example_answer_text.split())

            # TODO: filter here !!!

            if i == 0:
                head = "\t".join([
                    "pid", "sid", "question", "ans_sent", "answer",
                    "s_char_start", "s_char_end", "paragraph", "p_char_start", "p_char_end", "entailment_score",
                    "p_readibility", "p_perplexity", "p_length",
                    "s_readibility", "s_perplexity", "s_length",
                    "q_readibility", "q_perplexity", "q_length", "q_type", "q_type_id",
                    "a_readibility", "a_perplexity", "a_length", "a_chunk_tag", "a_ner_tag"])
                outfile.write(head + "\n")
            line_split += [
                paragraph_readibility, paragraph_perplexity, paragraph_length,
                ans_sent_readibility, ans_sent_perplexity, ans_sent_length,
                question_readibility, question_perplexity, question_length, question_type_text, question_type_id,
                answer_readibility, answer_perplexity, answer_length, answer_chunk_tag, answer_ner_tag]
            output_list = [str(item) for item in line_split]
            outfile.write("\t".join(output_list).rstrip().replace("\n", "\\n") + "\n")
            i = i + 1
    outfile.close()
    infile.close()


def main(args):
    start = datetime.now()
    evaluate_and_filter(args.input_file, args.input_augmented_pkl_file, args.output_file)
    print(("Time of data evaluation: {}").format(datetime.now() - start))


if __name__ == "__main__":
    # # readability
    # text = "how are you?"
    # print(get_readibility(text))
    # text = "I feel tired and want to sleep."
    # print(get_readibility(text))
    # text = "Understanding users and text plays the central role in many downstream tasks such as search and recommendation."
    # print(get_readibility(text))
    # text = "We present our techniques to construct user attention graph and tag documents with it in GIANT."
    # print(get_readibility(text))

    # perplexity
    a = ["i wrote a book, i wrote a book, i wrote a book, i wrote a book,i wrote a book, i wrote a book.",
         "i wrote a book.",
         "i wrote a book about the life of two young people who fall in love with each other."]
    print([get_perplexity(i) for i in a])

    # entailment score
    main(parser.parse_args())

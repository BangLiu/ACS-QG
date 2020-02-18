"""
Get or predict the valid question type(s).
"""
from collections import Counter
from .config import *
from .answer_selector import select_answers
from common.constants import NLP
from data_loader.FQG_data import get_raw_examples
from data_loader.FQG_data_utils import get_question_type
from util.file_utils import save


def get_answer_chunk_tag(context_text, answer_start, answer_end, stop_words=['the', 'a', 'an']):
    label = "UNK"
    real = context_text[answer_start:answer_end + 1]
    selected_answers = select_answers(context_text)
    for selected_answer_text, char_st, char_ed, _, label in selected_answers:
        if char_st == answer_start and char_ed == answer_end:
            return label
        # gap between real and selected_answer_text is stopword
        for wd in stop_words:
            if char_st + len(wd) + 1 == answer_start and selected_answer_text.startswith(wd + ' '):
                return label
            if answer_start + len(wd) + 1 == char_st and real.startswith(wd + ' '):
                return label
    return label


def get_answer_ner_tag(context, answer_text, processed_by_spacy=False):
    label = 'UNK'
    if not processed_by_spacy:
        doc = NLP(context)
    else:
        doc = context
    for ent in doc.ents:
        if ent.text == answer_text or answer_text in ent.text:
            return ent.label_
    return label


def get_answertag2qtype_mapping(answertag2qtype_dict_file, data_file, data_type):
    """
    Get the mapping between (answer_tags, potential question types).
    We either load a saved dictionary which we calculated and saved before,
    or we create such a dict by analyzing reference_file and save it for future usage.
    :param answertag2qtype_dict_file: we will save the result to this file.
    :param data_file: such as SQuAD data file. We use it to get the mapping.
    :param data_type: SQuAD or NewsQA. See get_raw_examples in FQG_data.py
    :return: a dict maps answer text tags (from the function get_answer_chunk_tags) to question types set.
    """
    examples = get_raw_examples(data_file, data_type)
    answertag2qtype = {}
    i = 0
    for e in examples:
        try:
            context_text = e["ans_sent"]
            answer_start = e["answer_start"]
            answer_text = e["answer_text"]
            answer_end = e["answer_start"] + len(answer_text) - 1
            question = e["question"]
            chunk_tag = get_answer_chunk_tag(context_text, answer_start, answer_end)
            ner_tag = get_answer_ner_tag(context_text, answer_text)
            answertag = "-".join([chunk_tag, ner_tag])
            qtype, qtype_id = get_question_type(question)
            if answertag in answertag2qtype:
                answertag2qtype[answertag].append(qtype)
            else:
                answertag2qtype[answertag] = [qtype]
        except:
            continue
        i = i + 1
        print(i)
        # if i > 20:
        #     break  # for debug

    answertag2qtype_set = {}
    answertag2qtype_counter = {}
    for k in answertag2qtype:
        answertag2qtype_set[k] = set(answertag2qtype[k])
        answertag2qtype_counter[k] = Counter(answertag2qtype[k])
    result = {"answertag2qtype": answertag2qtype,
              "answertag2qtype_set": answertag2qtype_set,
              "answertag2qtype_counter": answertag2qtype_counter}
    save(answertag2qtype_dict_file, result, message="save answertag2qtype dict")
    print(answertag2qtype_set)
    print(answertag2qtype_counter)
    return answertag2qtype_set


def select_question_types(context_text, answer_start, answer_end, answertag2qtype_set):
    """
    Given context and answer, we can get the valid question types.
    Return a set of valid question types.
    """
    answer_text = context_text[answer_start:answer_end + 1]
    chunk_tag = get_answer_chunk_tag(context_text, answer_start, answer_end)
    ner_tag = get_answer_ner_tag(context_text, answer_text)
    answertag = "-".join([chunk_tag, ner_tag])
    if answertag in answertag2qtype_set:
        return answertag2qtype_set[answertag]
    else:
        return []


if __name__ == "__main__":
    pass

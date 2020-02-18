"""
Given a sentence, sample (answer, clue, style) for it.
The basic idea is like the following.

P(a, c, s) = p(a) * p(c|a) * p(s|c, a)
           = p(a|a_tag, a_length) * p(c|c_tag, dep_dist) * p(s|a_tag)


# prepare
1. read the train dataset of squad1.1_zhou
2. for each line
       get a_tag, a_length, c_tag, dep_dist, s
3. get p(a|a_tag, a_length), p(c|c_tag, dep_dist), p(s|a_tag) by grid histogram


# sample
4. set each sentence no more than 5 answer, 2 type, 2 clue
5. for each new sentence
       perform parsing

       get all possible answers
       calculate answer prob and normalize
       sample 5 different answer for maximum 5 * 4 times. record answer tag, length, etc

       for each answer
           get all possible clue chunks
           calculate conditional clue probs, normalize.
           sample 2 clues for maximum 2 * 4 times. record clue info

           for each clue
              sample 2 q-types for maximum 2 * 4 times
"""
import copy
import math
import numpy as np
from collections import Counter
from .config import *
from common.constants import NLP, PARSER, FUNCTION_WORDS_LIST, QUESTION_TYPES, EXP_PLATFORM
from util.file_utils import save, load
from util.list_utils import weighted_sample
from util.prepro_utils import get_token_char_level_spans
from data_loader.FQG_data_utils import get_question_type
from data_loader import FQG_data

if EXP_PLATFORM.lower() == "venus":
    from nltk import data
    data.path.append('./nltk_need/nltk_data/')
import nltk


NOT_BEGIN_TOKENS_FOR_ANSWER_CLUE = [
    'of', 'for', 'to', 'is', 'are', 'and', 'was', 'were',
    ',', '?', ';', '!', '.']  # TODO: maybe more tokens


def get_token2char(doc):
    token2idx = {}
    idx2token = {}
    for token in doc:
        token2idx[token.i] = token.idx, token.idx + len(token.text) - 1
        for i in range(token.idx, token.idx + len(token.text)):
            idx2token[i] = token.i
    return token2idx, idx2token


def str_find(text, tklist):
    str_tk = ''
    for tk in tklist:
        str_tk += tk
    tk1 = tklist[0]
    pos = text.find(tk1)
    while pos < len(text) and pos >= 0:
        i, j = pos, 0
        while i < len(text) and j < len(str_tk):
            if text[i] == ' ':
                i += 1
                continue
            if text[i] == str_tk[j]:
                i += 1
                j += 1
            else:
                break
        if j == len(str_tk):
            return pos, i - 1
        newpos = text[pos + 1:].find(tk1)
        if newpos >= 0:
            pos = pos + 1 + newpos
        else:
            break
    return -1, -1


def _navigate(node):
    if type(node) is not nltk.Tree:
        return 1, 1, [('word', node)]
    # add position info of this node
    for idx, _ in enumerate(node.leaves()):
        tree_location = node.leaf_treeposition(idx)
        non_terminal = node[tree_location[:-1]]
        non_terminal[0] = non_terminal[0] + "___" + str(idx)

    max_depth = 0
    word_num = 0
    chunklist = []
    for child in node:
        child_depth, child_num, child_chunklist = _navigate(child)
        max_depth = max(max_depth, child_depth + 1)
        word_num += child_num
        chunklist += child_chunklist
    cur_node_chunk = [(node.label(), node.leaves())]
    chunklist += cur_node_chunk
    return max_depth, word_num, chunklist


def _dfs(doc, doc_token_list, cur_id, cur_path, max_depth, related):
    if len(cur_path) > max_depth:
        return
    if cur_id in related and len(related[cur_id]) <= len(cur_path):
        return
    related[cur_id] = cur_path
    for token in doc_token_list:
        if token.i != cur_id:
            continue
        new_path = copy.deepcopy(cur_path)
        try:
            new_path.append(token.dep_)
        except:
            continue
        _dfs(doc, doc_token_list, token.head.i, new_path, max_depth, related)
        for child in token.children:
            new_path = copy.deepcopy(cur_path)
            new_path.append(child.head.dep_)
            _dfs(doc, doc_token_list, child.i, new_path, max_depth, related)


def get_all_related(context_doc, doc_token_list):
    """
    Given a spaCy doc and its token list, get the dependency path between different tokens.
    The returned 3 results are like the following.
    idx2token: {0: Bob, 1: is, 2: eating, 3: a, 4: delicious, 5: cake, 6: in, 7: Vancouver, 8: .}
    idx2related: {0: [(0, []), (2, ['nsubj']), (1, ['nsubj', 'ROOT']), (5, ['nsubj', 'ROOT']), ...],
                  1: [(1, []), (2, ['aux']), (0, ['aux', 'ROOT']), (5, ['aux', 'ROOT']), ...],
                  ...,
                  8: ...}
    tokens: ['Bob', 'is', 'eating', 'a', 'delicious', 'cake', 'in', 'Vancouver', '.']
    """
    idx2token = {}
    idx2related = {}
    tokens = []
    for token in context_doc:
        idx2token[token.i] = token
        related = {}
        _dfs(context_doc, doc_token_list, token.i, [], len(context_doc) - 1, related)
        sort_related = sorted(related.items(), key=lambda x: len(x[1]))
        idx2related[token.i] = sort_related
        tokens.append(token.text)
    return idx2token, idx2related, tokens


def get_chunks(sentence, doc=None):
    """
    Input a sentence, output a list of its chunks (ner_tag, pos_tag, leaves_without_position, st, ed).
    Such as ('PERSON', 'NP', ['Beyoncé', 'Giselle', 'Knowles-Carter'], 0, 2).
    """
    tree = None
    try:
        tree = PARSER.parse(sentence)  # NOTICE: when sentence is too long, it will have error.
    except:
        pass
    if doc is None:
        doc = NLP(sentence)
    max_depth, node_num, orig_chunklist = _navigate(tree)

    chunklist = []
    # parse the result of _navigate
    for chunk in orig_chunklist:
        try:
            if chunk[0] == 'word':
                continue
            chunk_pos_tag, leaves = chunk
            leaves_without_position = []
            position_list = []
            for v in leaves:
                tmp = v.split('___')
                wd = tmp[0]
                index = int(tmp[1])
                leaves_without_position.append(wd)
                position_list.append(index)
            st = position_list[0]
            ed = position_list[-1]

            chunk_ner_tag = "UNK"
            chunk_text = " ".join(leaves_without_position)
            for ent in doc.ents:
                if ent.text == chunk_text or chunk_text in ent.text:
                    chunk_ner_tag = ent.label_

            chunklist.append((chunk_ner_tag, chunk_pos_tag, leaves_without_position, st, ed))
        except:
            continue

    return chunklist, tree, doc


def get_clue_info(question, sentence, answer, answer_start,
                  chunklist=None, y1_in_sent=None, doc=None, ques_doc=None, sent_limit=100):
    example = {
        "question": question,
        "ans_sent": sentence,
        "answer_text": answer}

    if doc is None:
        doc = NLP(sentence)

    if ques_doc is None:
        ques_doc = NLP(question)

    if chunklist is None:
        chunklist, _, _ = get_chunks(sentence, doc)

    example["ans_sent_tokens"] = [token.text for token in doc]
    example["ques_tokens"] = [token.text for token in ques_doc]
    example["ans_sent_doc"] = doc

    if y1_in_sent is None:
        spans = get_token_char_level_spans(sentence, example["ans_sent_tokens"])
        answer_end = answer_start + len(answer)
        answer_span = []
        for idx, span in enumerate(spans):
            if not (answer_end <= span[0] or
                    answer_start >= span[1]):
                answer_span.append(idx)
        y1_in_sent = answer_span[0]

    answer_start = y1_in_sent

    doc_token_list = [token for token in doc]
    idx2token, idx2related, context_tokens = get_all_related(doc, doc_token_list)

    clue_rank_scores = []
    for chunk in chunklist:
        candidate_clue = chunk[2]  # list of chunk words

        ques_lower = " ".join(example["ques_tokens"]).lower()
        candidate_clue_text = " ".join(candidate_clue).lower()
        ques_lemmas = [t.lemma_ for t in ques_doc]
        sent_lemmas = [t.lemma_ for t in doc]
        ques_tokens = [t.lower() for t in example["ques_tokens"]]
        candidate_clue_is_content = [int(w.lower() not in FUNCTION_WORDS_LIST) for w in candidate_clue]
        candidate_clue_lemmas = sent_lemmas[chunk[3]:chunk[4] + 1]
        candidate_clue_content_lemmas = [candidate_clue_lemmas[i] for i in range(len(candidate_clue_lemmas)) if candidate_clue_is_content[i] == 1]
        candidate_clue_lemmas_in_ques = [candidate_clue_lemmas[i] for i in range(len(candidate_clue_lemmas)) if candidate_clue_lemmas[i] in ques_lemmas]
        candidate_clue_content_lemmas_in_ques = [candidate_clue_content_lemmas[i] for i in range(len(candidate_clue_content_lemmas)) if candidate_clue_content_lemmas[i] in ques_lemmas]

        candidate_clue_tokens_in_ques = [candidate_clue[i] for i in range(len(candidate_clue)) if candidate_clue[i].lower() in ques_tokens]
        candidate_clue_content_tokens = [candidate_clue[i] for i in range(len(candidate_clue)) if candidate_clue_is_content[i] == 1]
        candidate_clue_content_tokens_in_ques = [candidate_clue_content_tokens[i] for i in range(len(candidate_clue_content_tokens)) if candidate_clue_content_tokens[i].lower() in ques_tokens]
        candidate_clue_content_tokens_in_ques_soft = candidate_clue_content_tokens_in_ques  # !!!! TODO: soft in.

        score = 0
        if (len(candidate_clue_lemmas_in_ques) == len(candidate_clue_lemmas) or len(candidate_clue_tokens_in_ques) == len(candidate_clue)) and \
                sum(candidate_clue_is_content) > 0 and \
                candidate_clue[0].lower() not in NOT_BEGIN_TOKENS_FOR_ANSWER_CLUE:
            score += len(candidate_clue_content_lemmas_in_ques)
            score += len(candidate_clue_content_tokens_in_ques)
            score += len(candidate_clue_content_tokens_in_ques_soft)
            score += int(candidate_clue_text in ques_lower)

        clue_rank_scores.append(score)
        # print("______".join([str(score), " ".join(chunk[2])]))  #!!!!!!!!! for debug

    if len(clue_rank_scores) == 0 or max(clue_rank_scores) == 0:
        clue_chunk = None
        clue_pos_tag = "UNK"
        clue_ner_tag = "UNK"
        clue_length = 0

        clue_answer_dep_path_len = -1

        selected_clue_binary_ids_padded = np.zeros([sent_limit], dtype=np.float32)
    else:
        clue_chunk = chunklist[clue_rank_scores.index(max(clue_rank_scores))]
        clue_pos_tag = clue_chunk[1]
        clue_ner_tag = clue_chunk[0]
        clue_length = clue_chunk[4] - clue_chunk[3] + 1

        clue_start = clue_chunk[3]
        clue_end = clue_chunk[4]
        clue_answer_dep_path_len = abs(clue_start - answer_start)
        answer_related = idx2related[answer_start]
        for tk_id, path in answer_related:
            if tk_id == clue_start:
                clue_answer_dep_path_len = len(path)

        selected_clue_binary_ids_padded = np.zeros([sent_limit], dtype=np.float32)
        if clue_start < sent_limit and clue_end < sent_limit:
            selected_clue_binary_ids_padded[clue_start:clue_end + 1] = 1

    clue_info = {
        "clue_pos_tag": clue_pos_tag,
        "clue_ner_tag": clue_ner_tag,
        "clue_length": clue_length,
        "clue_chunk": clue_chunk,
        "clue_answer_dep_path_len": clue_answer_dep_path_len,
        "selected_clue_binary_ids_padded": selected_clue_binary_ids_padded,
    }
    return clue_info


def get_answer_clue_style_info(sentence, question, answer, answer_start,
                               sent_limit=100, ques_limit=50, answer_limit=30, is_clue_topN=20):
    chunklist, tree, doc = get_chunks(sentence)

    # answer_info
    answer_pos_tag = "UNK"
    answer_ner_tag = "UNK"
    for chunk in chunklist:
        if answer == " ".join(chunk[2]):
            answer_ner_tag = chunk[0]
            answer_pos_tag = chunk[1]
            break
    answer_length = len(answer.split())

    # question style info
    question_type = get_question_type(question)

    # clue info
    clue_info = get_clue_info(
        question, sentence, answer, answer_start,
        chunklist=chunklist, y1_in_sent=None, doc=doc, ques_doc=None, sent_limit=sent_limit)

    example = {
        "question": question,
        "ans_sent": sentence,
        "answer_text": answer,
        "question_type": question_type,
        "answer_pos_tag": answer_pos_tag,
        "answer_ner_tag": answer_ner_tag,
        "answer_length": answer_length,
        "clue_pos_tag": clue_info["clue_pos_tag"],
        "clue_ner_tag": clue_info["clue_ner_tag"],
        "clue_answer_dep_path_len": clue_info["clue_answer_dep_path_len"],
        "clue_length": clue_info["clue_length"],
        "clue_chunk": clue_info["clue_chunk"]
    }

    return example


# run above function on squad data and save for plot, sample data, etc.
def get_dataset_info(filename, filetype, save_file=None,
                     sent_limit=100, ques_limit=50, answer_limit=30, is_clue_topN=20,
                     debug=False, debug_length=20):
    raw_examples = FQG_data.get_raw_examples(filename, filetype, debug, debug_length)
    examples_with_info = []
    for i in range(len(raw_examples)):
        print(i)
        e = raw_examples[i]
        sentence = e["ans_sent"]
        question = e["question"]
        answer = e["answer_text"]
        answer_start = e["answer_start"]
        new_e = get_answer_clue_style_info(
            sentence, question, answer, answer_start,
            sent_limit, ques_limit, answer_limit, is_clue_topN)
        examples_with_info.append(new_e)
        # print(new_e)  #!!!
        if debug and i >= debug_length:
            break
    if save_file is None:
        save_file = file_type + "_answer_clue_style_info.pkl"
    save(save_file, examples_with_info)
    return examples_with_info


def visualize_info_distribution():
    pass


def val2bin(input_val, min_val, max_val, bin_width):
    if input_val <= max_val and input_val >= min_val:
        return math.ceil((input_val - min_val) / bin_width)
    elif input_val > max_val:
        return math.ceil((max_val - min_val) / bin_width) + 1
    else:
        return -1


def get_sample_probs(filename, filetype, save_dataset_info_file=None, save_sample_probs_file=None,
                     sent_limit=100, ques_limit=50, answer_limit=30, is_clue_topN=20,
                     debug=False, debug_length=20,
                     answer_length_bin_width=3, answer_length_min_val=0, answer_length_max_val=30,
                     clue_dep_dist_bin_width=2, clue_dep_dist_min_val=0, clue_dep_dist_max_val=20): #!!!!!!!!!!!!! maybe 20 makes the prob be big for clue_dep > 20... set it as a big value like 100?
    """
    P(a, c, s) = p(a) * p(c|a) * p(s|c, a)
               = p(a|a_tag, a_length) * p(c|c_tag, dep_dist) * p(s|a_tag)
    """
    examples_with_info = get_dataset_info(
        filename, filetype, save_dataset_info_file,
        sent_limit, ques_limit, answer_limit, is_clue_topN,
        debug, debug_length)

    sla_tag = []  # for p(s|a_tag).  here we use "l" to denote "|"
    clc_tag_dep_dist = []  # for p(c|c_tag, dep_dist).  here we use "l" to denote "|"
    ala_tag_a_length = []  # for p(a|a_tag, a_length).  here we use "l" to denote "|"

    for e in examples_with_info:
        a_tag = "-".join([e["answer_pos_tag"], e["answer_ner_tag"]])  # answer tag
        s = e["question_type"][0]  # question style (type)
        a_length = e["answer_length"]
        a_length_bin = val2bin(a_length, answer_length_min_val, answer_length_max_val, answer_length_bin_width)
        c_tag = "-".join([e["clue_pos_tag"], e["clue_ner_tag"]])
        dep_dist = e["clue_answer_dep_path_len"]
        dep_dist_bin = val2bin(dep_dist, clue_dep_dist_min_val, clue_dep_dist_max_val, clue_dep_dist_bin_width)

        sla_tag.append("_".join([s, a_tag]))
        clc_tag_dep_dist.append("_".join([c_tag, str(dep_dist_bin)]))
        ala_tag_a_length.append("_".join([a_tag, str(a_length_bin)]))
    sla_tag = Counter(sla_tag)
    clc_tag_dep_dist = Counter(clc_tag_dep_dist)
    ala_tag_a_length = Counter(ala_tag_a_length)
    sample_probs = {
        "a": ala_tag_a_length,
        "c|a": clc_tag_dep_dist,
        "s|c,a": sla_tag}
    if save_sample_probs_file is None:
        save_sample_probs_file = filetype + "_sample_probs.pkl"
    save(save_sample_probs_file, sample_probs)
    # print(sample_probs)
    return sample_probs


def select_answers(sentence, sample_probs,
                   num_sample_answer=5,
                   answer_length_bin_width=3, answer_length_min_val=0, answer_length_max_val=30,
                   max_sample_times=20):
    # get all chunks
    chunklist, tree, doc = get_chunks(sentence)
    token2idx, idx2token = get_token2char(doc)

    # sample answer chunk
    chunk_ids = list(range(len(chunklist)))
    a_probs = []
    for chunk in chunklist:
        chunk_pos_tag = chunk[1]
        chunk_ner_tag = chunk[0]
        a_tag = "-".join([chunk_pos_tag, chunk_ner_tag])
        a_length = abs(chunk[3] - chunk[4] + 1)
        a_length_bin = val2bin(a_length, answer_length_min_val, answer_length_max_val, answer_length_bin_width)
        a_condition = "_".join([a_tag, str(a_length_bin)])  # condition of p(a|...)
        if a_condition in sample_probs["a"] and chunk[2][0].lower() not in NOT_BEGIN_TOKENS_FOR_ANSWER_CLUE:
            a_probs.append(sample_probs["a"][a_condition])
        else:
            a_probs.append(1)

    sampled_answer_chunk_ids = []
    sample_times = 0
    for sample_times in range(max_sample_times):
        sampled_chunk_id = weighted_sample(chunk_ids, a_probs)
        if sampled_chunk_id not in sampled_answer_chunk_ids:
            sampled_answer_chunk_ids.append(sampled_chunk_id)
        if len(sampled_answer_chunk_ids) >= num_sample_answer:
            break

    sampled_answers = []
    for chunk_id in sampled_answer_chunk_ids:
        chunk = chunklist[chunk_id]
        chunk_ner_tag, chunk_pos_tag, leaves, st, ed = chunk
        try:
            context = sentence
            char_st, char_ed = str_find(context, leaves)
            if char_st < 0:
                continue
            answer_text = context[char_st:char_ed + 1]
            st = idx2token[char_st]
            ed = idx2token[char_ed]
            answer_bio_ids = ['O'] * len(doc)
            answer_bio_ids[st: ed + 1] = ['I'] * (ed - st + 1)
            answer_bio_ids[st] = 'B'
            char_st = token2idx[st][0]
            char_ed = token2idx[ed][1]
            sampled_answers.append((answer_text, char_st, char_ed, st, ed, answer_bio_ids, chunk_pos_tag, chunk_ner_tag))
        except:
            continue

    return sampled_answers, chunklist, tree, doc


def select_question_types(sample_probs, selected_answer,
                          num_sample_style=2,
                          max_sample_times=20):
    (answer_text, char_st, char_ed, st, ed, answer_bio_ids, answer_pos_tag, answer_ner_tag) = selected_answer
    a_tag = "-".join([answer_pos_tag, answer_ner_tag])

    # get s probs
    styles = QUESTION_TYPES  # question types
    s_probs = []
    for s in QUESTION_TYPES:
        s_condition = "_".join([s, a_tag])
        if s_condition in sample_probs["s|c,a"]:
            s_probs.append(sample_probs["s|c,a"][s_condition])
        else:
            s_probs.append(1)

    # sample s
    sampled_styles = []
    sample_times = 0
    for sample_times in range(max_sample_times):
        sampled_s = weighted_sample(styles, s_probs)
        if sampled_s not in sampled_styles:
            sampled_styles.append(sampled_s)
        if len(sampled_styles) >= num_sample_style:
            break

    return sampled_styles


def select_clues(chunklist, doc, sample_probs, selected_answer,
                 num_sample_clue=2,
                 clue_dep_dist_bin_width=2, clue_dep_dist_min_val=0, clue_dep_dist_max_val=20,
                 max_sample_times=20):
    (answer_text, char_st, char_ed, st, ed, answer_bio_ids, answer_pos_tag, answer_ner_tag) = selected_answer

    doc_token_list = [token for token in doc]  # doc is sentence_doc
    idx2token, idx2related, context_tokens = get_all_related(doc, doc_token_list)

    # get c probs
    c_probs = []
    for chunk in chunklist:
        chunk_pos_tag = chunk[1]
        chunk_ner_tag = chunk[0]
        c_tag = "-".join([chunk_pos_tag, chunk_ner_tag])

        answer_start = st
        clue_start = chunk[3]
        clue_end = chunk[4]
        clue_answer_dep_path_len = abs(clue_start - answer_start)
        answer_related = idx2related[answer_start]
        for tk_id, path in answer_related:
            if tk_id == clue_start:
                clue_answer_dep_path_len = len(path)
        dep_dist = clue_answer_dep_path_len

        dep_dist_bin = val2bin(dep_dist, clue_dep_dist_min_val, clue_dep_dist_max_val, clue_dep_dist_bin_width)

        c_condition = "_".join([c_tag, str(dep_dist_bin)])  # condition of p(c|...)
        if c_condition in sample_probs["c|a"] and chunk[2][0].lower() not in NOT_BEGIN_TOKENS_FOR_ANSWER_CLUE:
            c_probs.append(sample_probs["c|a"][c_condition])
        else:
            c_probs.append(1)

    # sample c
    chunk_ids = list(range(len(chunklist)))
    sampled_clue_chunk_ids = []
    sample_times = 0
    for sample_times in range(max_sample_times):
        sampled_chunk_id = weighted_sample(chunk_ids, c_probs)
        if sampled_chunk_id not in sampled_clue_chunk_ids:
            sampled_clue_chunk_ids.append(sampled_chunk_id)
        if len(sampled_clue_chunk_ids) >= num_sample_clue:
            break

    sampled_clues = []
    for chunk_id in sampled_clue_chunk_ids:
        chunk = chunklist[chunk_id]
        clue_start = chunk[3]
        clue_end = chunk[4]
        clue_text = ' '.join(context_tokens[clue_start:clue_end + 1])
        clue_binary_ids = [0] * len(doc_token_list)
        clue_binary_ids[clue_start:clue_end + 1] = [1] * (clue_end - clue_start + 1)
        clue = {"clue_text": clue_text, "clue_binary_ids": clue_binary_ids}
        sampled_clues.append(clue)

    return sampled_clues


# # get SAMPLE_PROBS
# CURRENT_PATH = os.getcwd().split("/")
# DATA_PATH = "/".join(CURRENT_PATH[:-4]) + "/Datasets/"

# DATA_ACS_INFO_FILE_PATH = DATA_PATH + "processed/SQuAD1.1-Zhou/squad_ans_clue_style_info.pkl"
# SAMPLE_PROBS_FILE_PATH = DATA_PATH + "processed/SQuAD1.1-Zhou/squad_sample_probs.pkl"
# SQUAD_FILE = DATA_PATH + "original/SQuAD1.1-Zhou/train.txt"

# # !!!NOTICE: remember to clear these files when needed, otherwise we won't re-calculate.
# if not os.path.isfile(SAMPLE_PROBS_FILE_PATH):
#     print(SAMPLE_PROBS_FILE_PATH + " not exist.\nNow start generate these files.\n")
#     # if not exist, generate mapping dict and save to file
#     data_file = DATA_PATH + "original/SQuAD1.1-Zhou/train.txt"
#     data_type = "squad"
#     get_sample_probs(
#         filename=SQUAD_FILE, filetype="squad",
#         save_dataset_info_file=DATA_ACS_INFO_FILE_PATH, save_sample_probs_file=SAMPLE_PROBS_FILE_PATH,
#         sent_limit=100, ques_limit=50, answer_limit=30, is_clue_topN=20,
#         debug=False, debug_length=20,
#         answer_length_bin_width=3, answer_length_min_val=0, answer_length_max_val=30,
#         clue_dep_dist_bin_width=2, clue_dep_dist_min_val=0, clue_dep_dist_max_val=20)

# SAMPLE_PROBS = load(SAMPLE_PROBS_FILE_PATH)
# print(SAMPLE_PROBS_FILE_PATH + " loaded.\n")


def augment_qg_data(sentence, sample_probs,
                    num_sample_answer=5, num_sample_clue=2, num_sample_style=2,
                    answer_length_bin_width=3, answer_length_min_val=0, answer_length_max_val=30,
                    clue_dep_dist_bin_width=2, clue_dep_dist_min_val=0, clue_dep_dist_max_val=20,
                    max_sample_times=20):
    #!!! NOTICE: these default args must be the same with get_sample_probs:
    # answer_length_bin_width=3, answer_length_min_val=0, answer_length_max_val=30,
    # clue_dep_dist_bin_width=2, clue_dep_dist_min_val=0, clue_dep_dist_max_val=20,
    # A better way is to save these args together with SAMPLE_PROBS and do not pass them as arguments in
    # functions other than get_sample_probs.
    # Besides, these arguments should be set according to the visualization of dataset answer_clue_style info.
    # Well, currently we just won't touch it.
    sampled_infos = []

    # sample answer chunk
    sampled_answers, chunklist, tree, doc = select_answers(
        sentence, sample_probs,
        num_sample_answer,
        answer_length_bin_width, answer_length_min_val, answer_length_max_val,
        max_sample_times)

    for ans in sampled_answers:
        (answer_text, char_st, char_ed, st, ed, answer_bio_ids, answer_pos_tag, answer_ner_tag) = ans
        info = {"answer": {"answer_text": answer_text, "char_start": char_st, "char_end": char_ed,
                           "answer_bio_ids": answer_bio_ids, "answer_chunk_tag": answer_pos_tag},
                "styles": None,
                "clues": None}
        # sample style
        styles = select_question_types(sample_probs, ans,
                                       num_sample_style,
                                       max_sample_times)
        info["styles"] = list(styles)

        # sample clue
        clues = select_clues(chunklist, doc, sample_probs, ans,
                             num_sample_clue,
                             clue_dep_dist_bin_width, clue_dep_dist_min_val, clue_dep_dist_max_val,
                             max_sample_times)
        info["clues"] = clues

        sampled_infos.append(info)

    result = {"context": sentence, "selected_infos": sampled_infos, "ans_sent_doc": doc}
    return result


if __name__ == "__main__":
    # test get_chunks
    sentence = "Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress.\n"
    chunklist = get_chunks(sentence)
    print(chunklist)
    print("... above tests get_chunks\n")

    # test get_answer_clue_style_info
    sentence = "Bob is eating a delicious cake in Vancouver."
    question = "Where is Bob eating cake?"
    answer = "Vancouver"
    answer_start = 34
    example = get_answer_clue_style_info(sentence, question, answer, answer_start)
    print(example)
    print("... above tests get_answer_clue_style_info\n")

    # test get_sample_probs
    filename = "../../../../../Datasets/original/SQuAD1.1-Zhou/train.txt"
    filetype = "squad"
    save_dataset_info_file = "../../../../../Datasets/processed/SQuAD1.1-Zhou/squad_ans_clue_style_info.pkl"
    save_sample_probs_file = "../../../../../Datasets/processed/SQuAD1.1-Zhou/squad_sample_probs.pkl"
    sample_probs = get_sample_probs(
        filename, filetype, save_dataset_info_file, save_sample_probs_file,
        sent_limit=100, ques_limit=50, answer_limit=30, is_clue_topN=20,
        debug=True, debug_length=20,
        answer_length_bin_width=3, answer_length_min_val=0, answer_length_max_val=30,
        clue_dep_dist_bin_width=2, clue_dep_dist_min_val=0, clue_dep_dist_max_val=20)
    print("... above tests get_sample_probs\n")
    # NOTICE: we have run for non debug mode on squad. The files are saved to:
    # "../../../../../Datasets/processed/SQuAD1.1-Zhou/squad_ans_clue_style_info_full_train.pkl"
    # "../../../../../Datasets/processed/SQuAD1.1-Zhou/squad_sample_probs_full_train.pkl"
    # use these files to plot figures.

    # test weighted_sample
    result = [weighted_sample(["a", "b"], [0.5, 0.5]) for _ in range(30)]
    print(result)
    print("... above tests weighted_sample\n")

    # test augment_qg_data
    result = augment_qg_data("Bob is eating a delicious cake in Vancouver.", sample_probs)
    print("sampled result is:    ")
    print(result)
    print("... above tests augment_qg_data\n")

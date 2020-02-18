import numpy as np
import sys
from .config import *
from util.nlp_utils import get_synonyms, get_antonyms, get_semantic_related_words  # get_all_word_forms
from common.constants import NLP, INFO_QUESTION_TYPES, BOOL_QUESTION_TYPES, Q_TYPE2ID_DICT
from .content_separator import separate_content_function_words


def get_related_words(token, topN):
    """
    Given a word or spacy token, return its different types of
    related words as a dict.
    """
    # if isinstance(token, str):
    #     token = NLP(token)[0]  # NOTICE: a bug is that it may split a word into multiple tokens.
    # all_word_forms = get_all_word_forms(token.lower())
    # synonyms = get_synonyms(token)
    # antonyms = get_antonyms(token)
    semantic_related = get_semantic_related_words(token, topN)
    related_words = {}
    # related_words["all_word_forms"] = all_word_forms
    # related_words["synonyms"] = synonyms
    # related_words["antonyms"] = antonyms
    related_words["semantic_related"] = semantic_related
    return related_words


def get_related_words_dict(vocab, topN):
    """
    Given a vocab of words, return each word's different types of
    related words as a two-layered dict.
    TODO: make it parallel.
    """
    print("start get related words dict")
    related_words_dict = {}
    i = 0
    for token in vocab:
        print(i)
        related_words_dict[token] = get_related_words(token, topN)
        i += 1
    print("end get related words dict")
    return related_words_dict


def get_related_words_ids_mat(word2id_dict, topN):
    """
    Given a vocab of words, return each word's glove related words as a 2D matrix.
    """
    id2word_dict = dict([[v, k] for k, v in word2id_dict.items()])
    related_words_ids_mat = []
    for idx in range(len(id2word_dict)):
        token = id2word_dict[idx]
        related_words = get_semantic_related_words(token, topN)
        related_ids = [word2id_dict[w] if w in word2id_dict else -1 for w in related_words]
        related_ids_padded = [-1] * topN  # NOTICE: -1 is pad id to indicate non word
        related_ids_padded[:min(len(related_ids), topN)] = related_ids[:topN]
        related_words_ids_mat.append(related_ids_padded)
    return related_words_ids_mat


def get_related_words_ids_mat_with_related_words_dict(word2id_dict, topN, related_words_dict):
    """
    Given a vocab of words, return each word's glove related words as a 2D matrix
    with pre-calculated related_words_dict.
    """
    id2word_dict = dict([[v, k] for k, v in word2id_dict.items()])
    related_words_ids_mat = []
    for idx in range(len(id2word_dict)):
        token = id2word_dict[idx]
        if token in related_words_dict:
            related_words = related_words_dict[token]["semantic_related"]
        else:
            related_words = []  # NOTICE: do we need to always put itself as first related word?
        related_ids = [word2id_dict[w] if w in word2id_dict else -1 for w in related_words]
        related_ids_padded = [-1] * topN  # -1 is pad id to indicate non word
        related_ids_padded[:min(len(related_ids), topN)] = related_ids[:topN]
        related_words_ids_mat.append(related_ids_padded)
    return related_words_ids_mat


def get_softcopy_ids(input, output, sent_length, topN, related_words_dict=None):
    """
    Given input and output text, we get a padded soft copy target
    numpy array with a pre-calculated related_words_dict dict.
    """
    if isinstance(input, str):
        input = NLP(input)
    if isinstance(output, str):
        output = NLP(output)

    input_tokens = set([token.text.lower() for token in input])
    input_lemmas = set([token.lemma_ for token in input])
    input_synonyms = set()
    input_antomyms = set()
    input_semantic_related = set()
    for token in input:
        if related_words_dict is not None:
            synonyms = set(related_words_dict[token.text]["synonyms"])
            antonyms = set(related_words_dict[token.text]["antonyms"])
            semantic_related = set(related_words_dict[token.text]["semantic_related"][:topN])
        else:
            synonyms = set(get_synonyms(token.text))
            antonyms = set(get_antonyms(token.text))
            semantic_related = set(get_semantic_related_words(token.text, topN))
        input_synonyms = set.union(input_synonyms, synonyms)
        input_antomyms = set.union(input_antomyms, antonyms)
        input_semantic_related = set.union(input_semantic_related, semantic_related)

    is_copy_padded = np.zeros([sent_length], dtype=np.float32)
    i = 0
    for token in output:
        if token.text.lower() in input_tokens:  # hard copy
            is_copy_padded[i] = 1
        elif token.lemma_ in input_lemmas:  # variant form
            is_copy_padded[i] = 2
        elif token.text.lower() in input_synonyms:  # synonym
            is_copy_padded[i] = 3
        elif token.text.lower() in input_antomyms:  # antonym
            is_copy_padded[i] = 4
        elif token.text.lower() in input_semantic_related or token.text in input_semantic_related:  # semantic related
            is_copy_padded[i] = 5
        else:
            pass

        i += 1
        if i >= sent_length:
            break

    return is_copy_padded


def get_copy_labels(input_tokens, output_tokens, input_padded_length, output_padded_length, word2id_dict, related_words_ids_mat):
    input_tokens = [token.lower() for token in input_tokens]
    output_tokens = [token.lower() for token in output_tokens]

    # hard copy labels
    switch = np.zeros(output_padded_length, dtype=np.int32)  # 0/1 indicators of whether each token in output is copied from input
    copy_position = np.zeros(output_padded_length, dtype=np.int32)  # index of copy positions in input for copied output tokens
    # NOTICE: shall we make it -1 for pad copy position??? !!!!!!

    # NOTICE: this is because the original code use only oov copy. We keep it for comparison.
    switch_oov = np.zeros(output_padded_length, dtype=np.int32)  # 0/1 indicators of whether each oov token in output is copied from input
    copy_position_oov = np.zeros(output_padded_length, dtype=np.int32)  # index of copy positions in input for copied oov output tokens

    # soft copy labels
    switch_soft = np.zeros(output_padded_length, dtype=np.int32)  # 0/1 indicators of whether each token in output is soft copied from input
    copy_position_soft = np.zeros(output_padded_length, dtype=np.int32)  # index of copy positions in input for soft copied output tokens

    input_copied_hard_soft = np.zeros(input_padded_length, dtype=np.int32)

    # word indexes of input, output tokens
    src_id_list = []
    tgt = np.zeros(output_padded_length, dtype=np.int32)

    # get all input tokens' related word indexes
    for idx, w in enumerate(input_tokens):
        word_idx = None
        for each in (w, w.lower(), w.capitalize(), w.upper()):
            if each in word2id_dict:
                word_idx = word2id_dict[each]
                break
        src_id_list.append(word_idx if word_idx is not None else word2id_dict["<oov>"])
    input_related_word_ids = [related_words_ids_mat[idx] for idx in src_id_list]
    input_related_word_ids_flat = [item for sublist in input_related_word_ids for item in sublist]
    input_related_word_ids_set = set(input_related_word_ids_flat)

    # get copy labels and tgt word indexes
    for idx, tgt_word in enumerate(output_tokens):
        # get tgt index
        word_idx = None
        for each in (tgt_word, tgt_word.lower(), tgt_word.capitalize(), tgt_word.upper()):
            if each in word2id_dict:
                word_idx = word2id_dict[each]
                break
        tgt[idx] = word_idx if word_idx is not None else word2id_dict["<oov>"]

        # get hard copy features
        if tgt_word.lower() in input_tokens:  # NOTICE lower
            switch[idx] = 1
            copy_position[idx] = input_tokens.index(tgt_word.lower())  # NOTICE: here we haven't consider multiple same tokens
            input_copied_hard_soft[copy_position[idx]] = 1  # NOTICE: here we haven't consider multiple same tokens

        # get oov hard copy features
        if tgt_word.lower() in input_tokens and tgt[idx] == word2id_dict["<oov>"]:    # NOTICE lower
            switch_oov[idx] = 1
            copy_position_oov[idx] = input_tokens.index(tgt_word.lower())

        # get soft copy features
        if tgt_word.lower() not in input_tokens and \
                tgt[idx] in input_related_word_ids_set and \
                tgt[idx] != word2id_dict["<oov>"]:
            distance = [related_word_list.index(word_idx) + 1
                        if word_idx in related_word_list else sys.maxsize
                        for related_word_list in input_related_word_ids]
            copy_position_soft[idx] = distance.index(min(distance))  # NOTICE: here we haven't consider multiple same tokens
            # NOTICE: here we use add min distance. So that we can
            # change switch soft by threshold during training
            switch_soft[idx] = min(distance) + 1  # NOTICE: so it will start from 2. 2 means it is the most similar word by glove
            # NOTICE: here we haven't consider multiple same tokens
            # NOTICE: here we handle that input_copied_hard_soft[copy_position_soft[idx]] maybe overwritten by multiple output tokens
            # Because copy_position_soft[idx] may be the same for different idx
            if input_copied_hard_soft[copy_position_soft[idx]] == 0:
                input_copied_hard_soft[copy_position_soft[idx]] = switch_soft[idx]
            else:
                input_copied_hard_soft[copy_position_soft[idx]] = min(switch_soft[idx], input_copied_hard_soft[copy_position_soft[idx]])

    return tgt, switch, copy_position, switch_oov, copy_position_oov, switch_soft, copy_position_soft, input_copied_hard_soft


def get_clue_ids(context, context_is_content_ids_padded, question, sent_length, topN, related_words_dict=None):
    """
    Analysis sentence and given question, get that question's clue chunk[s].
    """
    is_copy_padded = get_softcopy_ids(question, context, sent_length, topN, related_words_dict)
    is_copy_padded = (is_copy_padded > 0).astype(float)

    # we set clue word must be a content word
    is_clue_padded = is_copy_padded
    if context_is_content_ids_padded is not None:
        is_clue_padded = (is_copy_padded + context_is_content_ids_padded == 2.0).astype(float)

    return is_clue_padded


def get_clue_ids_with_input_copied_hard_soft(input_copied_hard_soft_padded, context_is_content_ids_padded, topN):
    """
    :param input_copied_hard_soft_padded: numpy array.
        This indicates whether each input token is related to output.
        For example, if it is np.array([0, 0, 2, 4, 1, 0]), then
        0 means not related or pad value.
        1 means it is the same with one of output tokens.
        n >=2 means it is related to one of output tokens, and the glove distance is n - 1.
    """
    is_clue_padded = \
        (((input_copied_hard_soft_padded > 0).astype(float) +
         (input_copied_hard_soft_padded <= topN + 1).astype(float) +
         context_is_content_ids_padded) == 3.0).astype(float)
    return is_clue_padded


def get_content_ids(text, function_words_list, sent_length):
    """
    Get a padded binary numpy array to indicate which part of
    input text tokens are content tokens.
    """
    if isinstance(text, str):
        text = NLP(text)

    is_content = separate_content_function_words(text, function_words_list)
    is_content_padded = np.zeros([sent_length], dtype=np.float32)
    is_content_padded[:min(len(is_content), sent_length)] = is_content[:sent_length]

    return is_content_padded


def get_answer_ids(start, end, sent_length):
    """
    Given start and end token index, get a padded binary array to indicate
    which part of input text tokens are answer tokens.
    """
    assert start <= end
    assert end < sent_length
    is_answer_padded = np.zeros([sent_length], dtype=np.float32)
    if start < end:
        for i in range(start, end + 1):
            is_answer_padded[i] = 1
    return is_answer_padded


def get_question_type(question):
    """
    Given a string question, return its type name and type id.
    :param question: question string.
    :return: (question_type_str, question_type_id)
    """
    words = question.split()
    for word in words:
        for i in range(len(INFO_QUESTION_TYPES)):
            if INFO_QUESTION_TYPES[i].upper() in word.upper():
                return (INFO_QUESTION_TYPES[i],
                        Q_TYPE2ID_DICT[INFO_QUESTION_TYPES[i]])
    for i in range(len(BOOL_QUESTION_TYPES)):
        if BOOL_QUESTION_TYPES[i].upper() == words[0].upper():
            return ("Boolean", Q_TYPE2ID_DICT["Boolean"])
    return ("Other", Q_TYPE2ID_DICT["Other"])

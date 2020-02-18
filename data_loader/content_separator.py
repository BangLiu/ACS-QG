"""
Input a sentence, we want to know which part of the sentence
words are function words, and the rest are content words.
"""
from itertools import chain
from .config import *
from util.prepro_utils import get_token_char_level_spans
from common.constants import AC_AUTOMATON, NLP, FUNCTION_WORDS_LIST


def _remove_overlap_spans(all_found_expression_spans):
    # check overlap spans, only keep the longest span if two overlap
    idx_of_spans_to_remove = []
    for i in range(len(all_found_expression_spans) - 1):
        if i in idx_of_spans_to_remove:
            continue
        for j in range(i + 1, len(all_found_expression_spans)):
            if j in idx_of_spans_to_remove:
                continue
            a = all_found_expression_spans[i]
            b = all_found_expression_spans[j]
            if len(set(range(a[0], a[1] + 1)).intersection(range(b[0], b[1] + 1))) > 0:
                len_a = a[1] - a[0]
                len_b = b[1] - b[0]
                if len_a > len_b:
                    idx_of_spans_to_remove.append(j)
                else:
                    idx_of_spans_to_remove.append(i)
    all_found_expression_spans_no_overlap = []
    for i in range(len(all_found_expression_spans)):
        if i not in idx_of_spans_to_remove:
            all_found_expression_spans_no_overlap.append(all_found_expression_spans[i])
    return all_found_expression_spans_no_overlap


def _get_fixed_expression_chunk_token_spans(sentence):
    tokens = sentence.split()
    token_spans = get_token_char_level_spans(sentence, tokens)
    all_found_expression_spans = []
    for item in AC_AUTOMATON.iter(sentence):
        # print(item)
        end_char_idx = item[0] + 1
        start_char_idx = item[0] + 1 - len(item[1][1])
        token_span = []
        for i in range(len(token_spans)):
            if token_spans[i][0] == start_char_idx:
                token_span.append(i)
            if token_spans[i][1] == end_char_idx:
                token_span.append(i)
            if len(token_span) == 2:
                break
        if len(token_span) != 2:  # there may have partial match: for example, "know that" matches with "now that"
            continue
        all_found_expression_spans.append(token_span)

    return all_found_expression_spans


def get_fixed_expression_chunk_token_ids(spacy_doc):
    tokens = [token.text for token in spacy_doc]
    lemmas = [token.lemma_ for token in spacy_doc]
    sentence = " ".join(tokens)
    sentence_lemma = " ".join(lemmas)
    fixed_expression_spans = _get_fixed_expression_chunk_token_spans(sentence)
    fixed_expression_spans += _get_fixed_expression_chunk_token_spans(sentence_lemma)
    fixed_expression_spans = [list(t) for t in set(tuple(element) for element in fixed_expression_spans)]
    fixed_expression_spans = _remove_overlap_spans(fixed_expression_spans)
    fixed_expression_chunk_token_ids = [list(range(span[0], span[1] + 1)) for span in fixed_expression_spans]
    return fixed_expression_chunk_token_ids


def separate_content_function_words(spacy_doc, function_words_list):
    """
    1. we merge stop words, function words/phrases lists to get FUNCTION_WORDS_LIST
    2. we get chunks of spacy_doc
    3. for each word or chunk, if it is inside the FUNCTION_WORDS_LIST,
       we tag the word or whole chunk as function words.
       the rest are content words.
    """
    all_chunk_token_ids = []

    # get noun chunks by spacy
    for chunk in spacy_doc.noun_chunks:
        chunk_token_ids = []
        for token in chunk:
            chunk_token_ids.append(token.i)
        all_chunk_token_ids.append(chunk_token_ids)

    # get fixed expression chunks by fixed expression list
    fixed_expression_chunk_token_ids = get_fixed_expression_chunk_token_ids(spacy_doc)
    to_remove = []
    for i in range(len(fixed_expression_chunk_token_ids)):
        expression_chunk_i = fixed_expression_chunk_token_ids[i]
        for j in range(len(all_chunk_token_ids)):
            noun_chunk_j = all_chunk_token_ids[j]
            if (len(set(expression_chunk_i).intersection(set(noun_chunk_j)))) > 0:
                to_remove.append(i)  # if overlap, we keep noun_chunk, remove expression chunk
                continue

    expression_chunk_token_ids = []
    for k in range(len(fixed_expression_chunk_token_ids)):
        if k not in to_remove:
            expression_chunk_token_ids.append(fixed_expression_chunk_token_ids[k])

    all_chunk_token_ids += expression_chunk_token_ids

    flat_list = list(chain.from_iterable(all_chunk_token_ids))

    split_ids = []
    i = 0
    while i < len(spacy_doc):
        if i not in flat_list:
            split_ids.append([i])
            i += 1
        else:
            for chunk_ids in all_chunk_token_ids:
                if i in chunk_ids:
                    split_ids.append(chunk_ids)
                    i = max(chunk_ids) + 1

    # get is_func ids
    is_content = []
    for chunk_ids in split_ids:
        text = " ".join([spacy_doc[id].text for id in chunk_ids])
        if text.lower() in function_words_list:
            is_content.extend([0] * len(chunk_ids))
        else:
            is_content.extend([1] * len(chunk_ids))

    # result1 = []
    # result2 = []
    # result3 = []
    # for i in range(len(spacy_doc)):
    #     if is_func[i] == 0:  # content
    #         result1.append(spacy_doc[i].ent_type_)
    #         result2.append(spacy_doc[i].pos_)
    #         result3.append("_")
    #     else:
    #         result1.append(spacy_doc[i].text)
    #         result2.append(spacy_doc[i].text)
    #         result3.append(spacy_doc[i].text)
    # print(" ".join(result1))
    # print(" ".join(result2))
    # print(" ".join(result3))
    return is_content


if __name__ == "__main__":
    # test_remove_overlap_spans():
    all_found_expression_spans = [[1, 3], [2, 4], [2, 5]]
    print(_remove_overlap_spans(all_found_expression_spans))

    # test content separator
    sentences = [
        "Mary has lived in England for ten years.",
        "He's going to fly to Chicago next week.",
        "I don't understand this chapter of the book.",
        "The children will be swimming in the ocean at five o'clock.",
        "John had eaten lunch before his colleague arrived.",
        "The best time to study is early in the morning or late in the evening.",
        "The trees along the river are beginning to blossom.",
        "Our friends called us yesterday and asked if we'd like to visit them next month.",
        "You'll be happy to know that she's decided to take the position.",
        "I won't give away your secret."]

    for sentence in sentences:
        print(sentence)
        spacy_doc = NLP(sentence)
        print(separate_content_function_words(spacy_doc, FUNCTION_WORDS_LIST))

# [[2, 5]]
# Mary has lived in England for ten years.
# [1. 0. 1. 0. 1. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0.]
# He's going to fly to Chicago next week.
# [0. 0. 0. 0. 1. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0.]
# I don't understand this chapter of the book.
# [0. 0. 0. 1. 1. 1. 0. 1. 1. 0. 0. 0. 0. 0. 0.]
# The children will be swimming in the ocean at five o'clock.
# [1. 1. 0. 0. 1. 0. 1. 1. 0. 1. 1. 0. 0. 0. 0.]
# John had eaten lunch before his colleague arrived.
# [1. 0. 1. 1. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0.]
# The best time to study is early in the morning or late in the evening.
# [1. 1. 1. 0. 1. 0. 1. 0. 1. 1. 0. 1. 0. 1. 1.]
# The trees along the river are beginning to blossom.
# [1. 1. 0. 1. 1. 0. 1. 0. 1. 0. 0. 0. 0. 0. 0.]
# Our friends called us yesterday and asked if we'd like to visit them next month.
# [1. 1. 1. 0. 1. 0. 1. 0. 0. 0. 0. 0. 1. 0. 1.]
# You'll be happy to know that she's decided to take the position.
# [0. 0. 0. 1. 0. 1. 0. 0. 0. 1. 0. 1. 1. 1. 0.]
# I won't give away your secret.
# [0. 0. 0. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0.]

"""
Functions that used in data preprocessing.
"""
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from .dict_utils import counter2ordered_dict
from common.constants import NLP, SPM
from allennlp.modules.elmo import batch_to_ids


def text2tokens(sentence):
    """
    Transform text to tokens list.
    :param sentence: input text
    :return: list of token texts
    """
    doc = NLP(sentence)
    return [token.text for token in doc]


def get_token_char_level_spans(text, tokens):
    """
    Get tokens' char-level [) spans in text.
    :param text: input text
    :param tokens: list of token texts
    :return: list of tokens' char-level [) spans in text,
             each span is a tuple
    """
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def spacydoc2tokens(doc):
    """
    Transform spaCy doc to tokens list.
    :param doc: spaCy doc
    :return: list of token texts
    """
    return [token.text for token in doc]


def word2wid(word, word2id_dict, OOV="<oov>"):
    """
    Transform single word to word index.
    :param word: a word
    :param word2id_dict: a dict map words to indexes
    :param OOV: a token that represents Out-of-Vocabulary words
    :return: int index of the word
    """
    for each in (word, word.lower(), word.capitalize(), word.upper()):
        if each in word2id_dict:
            return word2id_dict[each]
    return word2id_dict[OOV]


def char2cid(char, char2id_dict, OOV="<oov>"):
    """
    Transform single character to character index.
    :param char: a character
    :param char2id_dict: a dict map characters to indexes
    :param OOV: a token that represents Out-of-Vocabulary characters
    :return: int index of the character
    """
    if char in char2id_dict:
        return char2id_dict[char]
    return char2id_dict[OOV]


def spacydoc2wids(spacy_doc, word2id_dict, sent_length,
                  PAD="<pad>", OOV="<oov>"):
    """
    Transform spaCy doc to padded word indexes list.
    :param spacy_doc: a spaCy doc
    :param word2id_dict: a dict map words to indexes
    :param sent_length: maximum length (number of words) of input text
    :param PAD: a token that represents pad
    :param OOV: a token that represents Out-of-Vocabulary words
    :return: list of word indexes
    """
    word_ids = np.ones(sent_length, dtype=np.int32) * word2id_dict[PAD]
    words = [token.text for token in spacy_doc]
    for i, token in enumerate(words):
        if i == sent_length:
            break
        word_ids[i] = word2wid(token, word2id_dict, OOV)
    return word_ids


def spacydoc2cids(spacy_doc, char2id_dict, sent_length, word_length,
                  PAD="<pad>", OOV="<oov>"):
    """
    Transform spaCy doc to padded character indexes 2D list.
    :param spacy_doc: a spaCy doc
    :param char2id_dict: a dict map characters to indexes
    :param sent_length: maximum length (number of words) of input text
    :param word_length: maximum length (number of characters) of words
    :param PAD: a token that represents pad
    :param OOV: a token that represents Out-of-Vocabulary words
    :return: 2D list of character indexes
    """
    char_ids = np.ones(
        [sent_length, word_length], dtype=np.int32) * char2id_dict[PAD]
    words = [token.text for token in spacy_doc]
    chars = [list(token) for token in words]
    for i, token in enumerate(chars):
        if i == sent_length:
            break
        for j, char in enumerate(token):
            if j == word_length:
                break
            char_ids[i, j] = char2cid(char, char2id_dict, OOV)
    return char_ids


def spacydoc2tagids(spacy_doc, tag_type, tag2id_dict, sent_length,
                    PAD="<pad>", OOV="<oov>"):
    """
    Transform spaCy doc to padded tag indexes list.
    :param spacy_doc: a spaCy doc
    :param tag_type: type of tag, currently support "pos", "ner", "iob", "dep"
    :param tag2id_dict: a dict map words to tags
    :param sent_length: maximum length (number of words) of input text
    :param PAD: a token that represents pad
    :param OOV: a token that represents Out-of-Vocabulary words
    :return: list of tag indexes
    """
    tag_ids = np.ones(sent_length, dtype=np.int32) * tag2id_dict[PAD]
    if tag_type == "pos":
        tags = [token.tag_ for token in spacy_doc]
    elif tag_type == "ner":
        tags = [token.ent_type_ for token in spacy_doc]
    elif tag_type == "iob":
        tags = [token.ent_iob_ for token in spacy_doc]
    elif tag_type == "dep":
        tags = [token.dep_ for token in spacy_doc]
    else:
        print("tag_type must be POS, NER, IOB or DEP.")
    for i, token in enumerate(tags):
        if i == sent_length:
            break
        tag_ids[i] = char2cid(token, tag2id_dict, OOV)
    return tag_ids


def spacydoc2features(spacy_doc, feature_type, sent_length):
    """
    Transform spaCy doc to padded tokens feature list.
    :param spacy_doc: a spaCy doc
    :param feature_type: type of features
    :param sent_length: maximum length (number of words) of input text
    :return: list of token feature values
    """
    features_padded = np.zeros([sent_length], dtype=np.float32)
    if feature_type == "is_alpha":
        features = [float(token.is_alpha) for token in spacy_doc]
    elif feature_type == "is_ascii":
        features = [float(token.is_ascii) for token in spacy_doc]
    elif feature_type == "is_digit":
        features = [float(token.is_digit) for token in spacy_doc]
    elif feature_type == "is_lower":
        features = [float(token.is_lower) for token in spacy_doc]
    elif feature_type == "is_title":
        features = [float(token.is_title) for token in spacy_doc]
    elif feature_type == "is_punct":
        features = [float(token.is_punct) for token in spacy_doc]
    elif feature_type == "is_left_punct":
        features = [float(token.is_left_punct) for token in spacy_doc]
    elif feature_type == "is_right_punct":
        features = [float(token.is_right_punct) for token in spacy_doc]
    elif feature_type == "is_bracket":
        features = [float(token.is_bracket) for token in spacy_doc]
    elif feature_type == "is_quote":
        features = [float(token.is_quote) for token in spacy_doc]
    elif feature_type == "is_currency":
        features = [float(token.is_currency) for token in spacy_doc]
    elif feature_type == "is_stop":
        features = [float(token.is_stop) for token in spacy_doc]
    elif feature_type == "like_url":
        features = [float(token.like_url) for token in spacy_doc]
    elif feature_type == "like_num":
        features = [float(token.like_num) for token in spacy_doc]
    elif feature_type == "like_email":
        features = [float(token.like_email) for token in spacy_doc]
    else:
        print("Incorrect feature type.")
    features_padded[:min(len(features), sent_length)] = features[:sent_length]
    return features_padded


def spacydoc2is_overlap(spacy_doc, token_set, sent_length, lower=False):
    """
    Transform spaCy doc to a list of 1/0, where 1 indicates the token
    shows up in a token_set, and 0 means it doesn't show up in token_set.
    :param spacy_doc: a spaCy doc
    :param token_set: a set of tokens
    :param sent_length: maximum length (number of words) of input text
    :return: list of token overlap feature values
    """
    if lower:
        token_set = [token.lower() for token in token_set]
    is_overlap_padded = np.zeros([sent_length], dtype=np.float32)
    if lower:
        is_overlap = [float(token.text.lower() in token_set) for token in spacy_doc]
    else:
        is_overlap = [float(token.text in token_set) for token in spacy_doc]
    is_overlap_padded[:min(len(is_overlap), sent_length)] = \
        is_overlap[:sent_length]
    return is_overlap_padded


def feature2ids(feature, feature2id_dict, original_length, sent_length,
                PAD="<pad>", OOV="<oov>"):
    feature_ids = np.ones(sent_length, dtype=np.int32) * feature2id_dict[PAD]
    for i, feat in enumerate(feature):
        if i == min(sent_length, original_length):
            break
        feature_ids[i] = char2cid(feat, feature2id_dict, OOV)
    return feature_ids


def get_answer_iob(original_length, start, end):
    assert start <= end
    assert end < original_length
    answer_iob = ['O'] * original_length
    answer_iob[start] = 'B'
    if start < end:
        for i in range(start + 1, end + 1):
            answer_iob[i] = 'I'
    return answer_iob


def get_embedding(counter, data_type,
                  emb_file=None, size=None, vec_size=None,
                  limit=-1, specials=["<pad>", "<oov>", "<sos>", "<eos>"]):
    """
    Get embedding matrix and dict that maps tokens to indexes.
    :param counter: a Counter object that counts different tokens
    :param data_type: a string name of data type
    :param emb_file: file of embeddings, such as Glove file
    :param size: number of different tokens
    :param vec_size: dimension of embedding vectors
    :param limit: filter low frequency tokens with freq < limit
    :param specials: list of special tokens
    :return: emb_mat
                 2D list of embedding matrix
             token2idx_dict
                 dict that maps tokens to indexes
    NOTICE: here we put <pad> at index 0 is the best. Otherwise, some other
    code may have problem.
    """
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}

    # get sorted word counter: ordered dict
    counter = counter2ordered_dict(counter)
    filtered_elements = [k for k, v in counter.items()
                         if (v > limit and k not in specials)]

    # get embedding_dict: (word, vec) pairs
    current_word_idx = 0
    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh):
                if current_word_idx == size - len(specials):
                    break
                array = line.split()
                if len(array) < vec_size:
                    continue
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in filtered_elements:
                    embedding_dict[word] = vector
                    current_word_idx += 1
    else:
        assert vec_size is not None
        for token in filtered_elements:
            if size is not None and current_word_idx == size - len(specials):
                break
            embedding_dict[token] = [
                np.random.normal(scale=0.1) for _ in range(vec_size)]
            current_word_idx += 1

    # get token2idx_dict: (word, index) pairs
    token2idx_dict = {}
    nid = 0
    for token in counter:
        if token in embedding_dict:
            token2idx_dict[token] = nid + len(specials)
            nid += 1
    for i in range(len(specials)):
        token2idx_dict[specials[i]] = i
        embedding_dict[specials[i]] = [0. for _ in range(vec_size)]

    # get idx2emb_dict: (index, vec) pairs
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}

    # get emb_mat according to idx2emb_dict: num_words x emb_dim
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    print("{} / {} tokens have corresponding {} embedding vector".format(
        len(embedding_dict), len(filtered_elements) + len(specials),
        data_type))
    return emb_mat, token2idx_dict


def spacytoken2bpe(spacy_token):
    """
    Input spacy token, output a list of sub words.
    """
    text = spacy_token.text
    text = text.lower()
    if spacy_token.is_digit:
        text = 0
    if spacy_token.like_url:
        text = "<url>"
    return SPM.EncodeAsPieces(str(text))


def spacydoc2bpe(spacy_doc):
    """
    Input spacy doc, output a 2d list of sub words.
    Each list is the sub words of a token.
    """
    bpe = []
    for token in spacy_doc:
        text = token.text
        text = text.lower()
        if token.is_digit:
            text = 0
        if token.like_url:
            text = "<url>"
        bpe.append(SPM.EncodeAsPieces(str(text)))
    return bpe


def spacydoc2bpeids(spacy_doc, bpe2id_dict, sent_length, word_bpe_length,
                    PAD="<pad>", OOV="<oov>"):
    bpe_ids = np.ones(
        [sent_length, word_bpe_length], dtype=np.int32) * bpe2id_dict[PAD]
    bpes = spacydoc2bpe(spacy_doc)
    for i, token in enumerate(bpes):
        if i == sent_length:
            break
        for j, bpe in enumerate(token):
            if j == word_bpe_length:
                break
            bpe_ids[i, j] = char2cid(bpe, bpe2id_dict, OOV)
    return bpe_ids


def init_counters(tags, not_count_tags):
    """
    Given a list of tags, such as ["word", "char", "ner", ...],
    return a dictionary of counters: tag -> Counter instance.
    """
    counters = {}
    for tag in tags:
        counters[tag] = Counter()
    for tag in not_count_tags:
        counters[tag] = Counter()
        for val in not_count_tags[tag]:
            counters[tag][val] += 1e30
    return counters


def update_counters(counters, spacy_doc, tags, increment=1):
    """
    Given a spacy doc of a sentence, update a dictionary of counters.
    :param counters: the counters to update
    :param spacy_doc: a spacy doc of a sentence
    :param tags: a list of tags
    :param increment: the amount of count to add for each token
    :return: updated counters
    """
    for token in spacy_doc:
        if "word" in tags:
            counters["word"][token.text] += increment
        if "char" in tags:
            for char in token.text:
                counters["char"][char] += increment
        if "pos" in tags:
            counters["pos"][token.tag_] += increment
        if "ner" in tags:
            counters["ner"][token.ent_type_] += increment
        if "iob" in tags:
            counters["iob"][token.ent_iob_] += increment
        if "dep" in tags:
            counters["dep"][token.dep_] += increment
        if "is_alpha" in tags:
            counters["is_alpha"][float(token.is_alpha)] += increment
        if "is_ascii" in tags:
            counters["is_ascii"][float(token.is_ascii)] += increment
        if "is_digit" in tags:
            counters["is_digit"][float(token.is_digit)] += increment
        if "is_lower" in tags:
            counters["is_lower"][float(token.is_lower)] += increment
        if "is_punct" in tags:
            counters["is_punct"][float(token.is_punct)] += increment
        if "is_bracket" in tags:
            counters["is_bracket"][float(token.is_bracket)] += increment
        if "is_stop" in tags:
            counters["is_stop"][float(token.is_stop)] += increment
        if "is_currency" in tags:
            counters["is_currency"][float(token.is_currency)] += increment
        if "is_quote" in tags:
            counters["is_quote"][float(token.is_quote)] += increment
        if "like_url" in tags:
            counters["like_url"][float(token.like_url)] += increment
        if "like_num" in tags:
            counters["like_num"][float(token.like_num)] += increment
        if "like_email" in tags:
            counters["like_email"][float(token.like_email)] += increment
        if "is_left_punct" in tags:
            counters["is_left_punct"][float(token.is_left_punct)] += increment
        if "is_right_punct" in tags:
            counters["is_right_punct"][
                float(token.is_right_punct)] += increment
        if "bpe" in tags:
            bpes = spacytoken2bpe(token)
            for bpe in bpes:
                counters["bpe"][bpe] += increment
    return counters


def tokens2ELMOids(tokens, sent_length):
    """
    Transform input tokens to elmo ids.
    :param tokens: a list of words.
    :param sent_length: padded sent length.
    :return: numpy array of elmo ids, sent_length * 50
    """
    elmo_ids = batch_to_ids([tokens]).squeeze(0)
    pad_c = (0, 0, 0, sent_length - elmo_ids.size(0))  # assume PAD_id = 0
    elmo_ids = torch.nn.functional.pad(elmo_ids, pad_c, value=0)
    elmo_ids = elmo_ids.data.cpu().numpy()
    return elmo_ids


def get_dependency_tree_edges(spacy_doc):
    # get spacy's dependency tree edges
    edges = []
    for token in spacy_doc:
        e = [token.head.i, token.i, token.dep_]
        edges.append(e)
    return edges


if __name__ == "__main__":
    text = "<sos> How are you? <eos>"
    text2 = "Apple is how are you three"
    counters = {}
    counters["word"] = Counter()
    counters["char"] = Counter()
    counters["pos"] = Counter()
    counters["ner"] = Counter()
    counters["iob"] = Counter()
    counters["dep"] = Counter()
    spacy_doc = NLP(text)
    spacy_doc2 = NLP(text2)
    print(spacydoc2bpe(spacy_doc2))
    for token in spacy_doc:
        counters["word"][token.text] += 1
        for char in token.text:
            counters["char"][char] += 1
        counters["pos"][token.tag_] += 1
        counters["ner"][token.ent_type_] += 1
        counters["iob"][token.ent_iob_] += 1
        counters["dep"][token.dep_] += 1
    emb_mats = {}
    emb_dicts = {}
    emb_mats["word"], emb_dicts["word"] = get_embedding(
        counters["word"], "word", vec_size=3)
    emb_mats["char"], emb_dicts["char"] = get_embedding(
        counters["char"], "char", vec_size=3)
    emb_mats["pos"], emb_dicts["pos"] = get_embedding(
        counters["pos"], "pos", vec_size=3)
    emb_mats["ner"], emb_dicts["ner"] = get_embedding(
        counters["ner"], "ner", vec_size=3)
    emb_mats["iob"], emb_dicts["iob"] = get_embedding(
        counters["iob"], "iob", vec_size=3)
    emb_mats["dep"], emb_dicts["dep"] = get_embedding(
        counters["dep"], "dep", vec_size=3)

    print("emb_dicts", emb_dicts)
    sent_length = 10
    word_length = 4
    token_set2 = set([token.text for token in spacy_doc2])
    print("spacydoc2is_overlap", spacydoc2is_overlap(
        spacy_doc, token_set2, sent_length))
    print("spacydoc2tokens", spacydoc2tokens(spacy_doc))
    print("spacydoc2wids", spacydoc2wids(
        spacy_doc, emb_dicts["word"], sent_length))
    print("spacydoc2cids", spacydoc2cids(
        spacy_doc, emb_dicts["char"], sent_length, word_length))
    poss = [token.tag_ for token in spacy_doc]
    ners = [token.ent_type_ for token in spacy_doc]
    iobs = [token.ent_iob_ for token in spacy_doc]
    deps = [token.dep_ for token in spacy_doc]
    print("poss", poss)
    print("ners", ners)
    print("iobs", iobs)
    print("deps", deps)
    print("spacydoc2posids", spacydoc2tagids(
        spacy_doc, "pos", emb_dicts["pos"], sent_length))
    print("spacydoc2nerids", spacydoc2tagids(
        spacy_doc, "ner", emb_dicts["ner"], sent_length))
    print("spacydoc2iobids", spacydoc2tagids(
        spacy_doc, "iob", emb_dicts["iob"], sent_length))
    print("spacydoc2depids", spacydoc2tagids(
        spacy_doc, "dep", emb_dicts["dep"], sent_length))
    print("spacydoc2is_alpha", spacydoc2features(
        spacy_doc, "is_alpha", sent_length))
    print("spacydoc2is_ascii", spacydoc2features(
        spacy_doc, "is_ascii", sent_length))
    print("spacydoc2is_digit", spacydoc2features(
        spacy_doc, "is_digit", sent_length))
    print("spacydoc2is_lower", spacydoc2features(
        spacy_doc, "is_lower", sent_length))
    print("spacydoc2is_punct", spacydoc2features(
        spacy_doc, "is_punct", sent_length))
    print("spacydoc2is_title", spacydoc2features(
        spacy_doc, "is_title", sent_length))
    print("spacydoc2is_left_punct", spacydoc2features(
        spacy_doc, "is_left_punct", sent_length))
    print("spacydoc2is_right_punct", spacydoc2features(
        spacy_doc, "is_right_punct", sent_length))
    print("spacydoc2is_bracket", spacydoc2features(
        spacy_doc, "is_bracket", sent_length))
    print("spacydoc2is_quote", spacydoc2features(
        spacy_doc, "is_quote", sent_length))
    print("spacydoc2is_currency", spacydoc2features(
        spacy_doc, "is_currency", sent_length))
    print("spacydoc2is_stop", spacydoc2features(
        spacy_doc, "is_stop", sent_length))
    print("spacydoc2like_url", spacydoc2features(
        spacy_doc, "like_url", sent_length))
    print("spacydoc2like_num", spacydoc2features(
        spacy_doc, "like_num", sent_length))
    print("spacydoc2like_email", spacydoc2features(
        spacy_doc, "like_email", sent_length))

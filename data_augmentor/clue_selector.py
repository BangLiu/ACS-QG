"""
Select the chunks that can be clue to given context and answer.
"""
import copy
from .config import *
from common.constants import NLP


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


def select_clues(context, answer, answer_bio_ids, max_dependency_distance, processed_by_spacy=False):
    """
    Select clue chunks given context and answer.
    """
    # return a list of [(clue_text, clue_binary_ids)] tuples.
    answer_start = answer_bio_ids.index('B')
    try:
        answer_end = list(reversed(answer_bio_ids)).index('I')
        answer_end = len(answer_bio_ids) - 1 - answer_end
    except:
        answer_end = answer_start
    if not processed_by_spacy:
        doc = NLP(context)
    else:
        doc = context

    doc_token_list = [token for token in doc]
    # text_str = ' '.join([tk.text for tk in doc])

    idx2token, idx2related, context_tokens = get_all_related(doc, doc_token_list)
    clue_flags = [0] * len(doc)
    for aid in range(answer_start, answer_end + 1):
        sort_related = idx2related[aid]
        for tk_id, path in sort_related:
            if (tk_id < answer_start or tk_id > answer_end) and len(path) <= max_dependency_distance:
                cur_clue = idx2token[tk_id]
                if cur_clue.pos_ not in ['ADP', 'DET', 'ADV', 'PUNCT', 'PART']:
                    clue_flags[tk_id] = 1
    clues = []
    i = 0
    while i < len(clue_flags):
        if clue_flags[i] == 0:
            i += 1
            continue
        j = i
        while j < len(clue_flags):
            if clue_flags[j] == 1:
                j += 1
            else:
                break
        clue_text = ' '.join(context_tokens[i:j])
        clue_binary_ids = [0] * len(clue_flags)
        clue_binary_ids[i:j] = [1] * (j - i)
        clues.append({"clue_text": clue_text, "clue_binary_ids": clue_binary_ids})
        i = j
    return clues


if __name__ == "__main__":
    context = "Mary has lived in England for ten years"
    answer = "England"
    answer_bio_ids = ["O", "O", "O", "O", "B", "O", "O", "O"]
    max_dependency_distance = 6
    clues = select_clues(context, answer, answer_bio_ids, max_dependency_distance)
    print(clues)  # [('Mary has lived', [1, 1, 1, 0, 0, 0, 0, 0]), ('ten years', [0, 0, 0, 0, 0, 0, 1, 1])]

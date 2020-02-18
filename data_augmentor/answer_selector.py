"""
Given input sentence, select chunks that can be answer.
Currently, we restrict answer to be a part of input sentence.
In the future, it can be directly generated from input sentence and other resources.
"""
import nltk
from .config import *
from common.constants import NLP, PARSER

ROOT = 'ROOT'


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


def _post(orig_chunklist):
    chunklist = []
    # parse the result of _navigate
    for chunk in orig_chunklist:
        try:
            if chunk[0] == 'word':
                continue
            label, leaves = chunk
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
            chunklist.append((label, leaves_without_position, st, ed))
        except:
            continue

    # delete PP which starts with [of, for]
    cand_chunks = []
    vp_list = []
    for chunk in chunklist:
        label, leaves, st, ed = chunk
        if label == 'VP':
            vp_list.append(chunk)
        else:
            if label == 'PP':
                if leaves[0] in ['of', 'OF', 'for', 'FOR', 'to', 'TO']:
                    continue
            cand_chunks.append(chunk)

    # delete VP which is not minimum chunk
    for chunk in vp_list:
        label, leaves, st, ed = chunk
        cur_str = ' '.join(leaves)
        contain_other = False
        for other_chunk in vp_list:
            other_leaves = other_chunk[1]
            if other_leaves == leaves:
                continue
            other_str = ' '.join(other_leaves)
            if cur_str.find(other_str) >= 0:
                contain_other = True
                break
        if not contain_other:
            cand_chunks.append(chunk)

    # other strategy of deleting chunk
    answer_chunks = []
    for chunk in chunklist:
        label, leaves, st, ed = chunk
        if label not in ['NP', 'VP', 'PP', 'UCP', 'ADVP']:
            continue
            pass
        answer_chunks.append(chunk)
    return answer_chunks


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


def select_answers(context, processed_by_spacy=False):
    """
    Input a context, we select which part of the input words belonging to the answer.
    """
    # return a list of [(answer_text, answer_bio_ids)] tuples.
    tree = None
    try:
        tree = PARSER.parse(context)  # TODO: if the context is too long, it will cause error.
    except:
        pass
    if not processed_by_spacy:
        doc = NLP(context)
    else:
        doc = context
    token2idx, idx2token = get_token2char(doc)
    max_depth, node_num, chunklist = _navigate(tree)
    answer_chunks = _post(chunklist)
    answers = []
    for chunk in answer_chunks:
        label, leaves, st, ed = chunk
        # print('leaves={}\tst={}\ted={}\tlabel={}'.format(leaves, st, ed, label))
        try:
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
            # print('answer_text={}\tchar_st={}\tchar_ed={}\tst={}\ted={}'.format(answer_text, char_st, char_ed, st, ed))
        except:
            continue
        answers.append((answer_text, char_st, char_ed, answer_bio_ids, label))

    return answers


if __name__ == "__main__":
    sent_list = [
        "Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress.\n"
    ]
    for sent in sent_list:
        answer_list = select_answers(sent)
        print("sent is: ", sent)
        print("answer list is: ", answer_list)

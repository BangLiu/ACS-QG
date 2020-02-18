"""
Given input sentence and answer, we get the potential clue pieces.
"""
import os
from .config import *
from .answer_selector import select_answers
from .clue_selector import select_clues
from .style_selector import get_answertag2qtype_mapping, select_question_types
from util.file_utils import load


# answer_tag mapping to question type set
current_path = os.getcwd().split("/")

DATA_PATH = "/".join(current_path[:-4]) + "/Datasets/"

ANSWERTAG2QTYPE_FILE_PATH = DATA_PATH + "processed/SQuAD1.1-Zhou/answertag2qtype_dict.pkl"
if not os.path.isfile(ANSWERTAG2QTYPE_FILE_PATH):
    # if not exist, generate mapping dict and save to file
    data_file = DATA_PATH + "original/SQuAD1.1-Zhou/train.txt"
    data_type = "squad"
    get_answertag2qtype_mapping(ANSWERTAG2QTYPE_FILE_PATH, data_file, data_type)


""" OUTPUT
{'NP-PERSON': {'How', 'Which', 'Boolean', 'When', 'Who', 'What', 'Where', 'Other'},
 'NP-UNK': {'How', 'Which', 'Boolean', 'When', 'Who', 'Why', 'What', 'Where', 'Other'},
 'NP-FAC': {'Which', 'When', 'Who', 'Why', 'What', 'Where', 'Other'},
 'VP-DATE': {'How', 'Which', 'When', 'Who', 'What', 'Where', 'Other'},
 'VP-UNK': {'How', 'Which', 'Boolean', 'When', 'Who', 'Why', 'What', 'Where', 'Other'},
 'VP-CARDINAL': {'How', 'Which', 'When', 'Who', 'What', 'Where', 'Other'},
 'NP-DATE': {'How', 'Which', 'Boolean', 'When', 'Who', 'What', 'Where', 'Other'},
 'NP-GPE': {'How', 'Which', 'Boolean', 'When', 'Who', 'Why', 'What', 'Where', 'Other'},
 'NP-ORG': {'How', 'Which', 'Boolean', 'When', 'Who', 'Why', 'What', 'Where', 'Other'},
 'NP-CARDINAL': {'How', 'Which', 'Boolean', 'When', 'Who', 'What', 'Where', 'Other'},
 'VP-ORG': {'How', 'Which', 'Boolean', 'When', 'Who', 'Why', 'What', 'Where', 'Other'},
 'VP-PERSON': {'How', 'Which', 'When', 'Who', 'Why', 'What', 'Where', 'Other'},
 'VP-WORK_OF_ART': {'How', 'Which', 'Who', 'Why', 'What', 'Where'},
 'NP-WORK_OF_ART': {'How', 'Which', 'When', 'Who', 'What', 'Where', 'Other'},
 'NP-PERCENT': {'How', 'Which', 'Boolean', 'When', 'Who', 'What', 'Where', 'Other'},
 'NP-QUANTITY': {'How', 'Which', 'When', 'What', 'Where'},
 'VP-ORDINAL': {'How', 'Which', 'When', 'What', 'Where'},
 'PP-ORG': {'What', 'Which', 'Where'},
 'PP-UNK': {'How', 'Which', 'Boolean', 'When', 'Who', 'Why', 'What', 'Where'},
 'VP-PERCENT': {'How', 'Which', 'When', 'What', 'Other'},
 'VP-GPE': {'How', 'Which', 'When', 'Who', 'What', 'Where', 'Other'},
 'NP-MONEY': {'What', 'How', 'When', 'Where'},
 'VP-MONEY': {'What', 'How', 'Where'},
 'VP-QUANTITY': {'How', 'Who', 'When', 'What'},
 'NP-LOC': {'How', 'Which', 'When', 'Who', 'What', 'Where', 'Other'},
 'VP-LOC': {'How', 'Which', 'When', 'Who', 'What', 'Where'},
 'VP-NORP': {'How', 'Which', 'Boolean', 'Who', 'What', 'Where'},
 'NP-NORP': {'How', 'Which', 'Boolean', 'When', 'Who', 'What', 'Where', 'Other'},
 'PP-PERSON': {'What', 'Who', 'Which'},
 'NP-PRODUCT': {'Which', 'When', 'Who', 'What', 'Where', 'Other'},
 'NP-EVENT': {'How', 'Which', 'When', 'Who', 'Why', 'What', 'Where', 'Other'},
 'PP-WORK_OF_ART': {'What', 'Which'},
 'VP-EVENT': {'How', 'Which', 'When', 'Who', 'What', 'Where'},
 'ADVP-WORK_OF_ART': {'What', 'Which'},
 'ADVP-UNK': {'How', 'Which', 'Boolean', 'When', 'Who', 'Why', 'What', 'Where', 'Other'},
 'VP-LAW': {'How', 'Which', 'When', 'Who', 'What', 'Where'},
 'VP-LANGUAGE': {'What', 'Which'},
 'NP-ORDINAL': {'How', 'Which', 'When', 'What', 'Where'},
 'VP-FAC': {'How', 'Which', 'When', 'Who', 'What', 'Where', 'Other'},
 'ADVP-ORDINAL': {'What', 'Which', 'Where'},
 'NP-LANGUAGE': {'What', 'Which', 'Boolean'},
 'PP-GPE': {'What', 'Who', 'Which', 'Where'},
 'ADVP-DATE': {'How', 'Which', 'When', 'What', 'Where'},
 'ADVP-NORP': {'What', 'Which'},
 'PP-CARDINAL': {'How', 'When', 'What'},
 'VP-PRODUCT': {'How', 'Which', 'Boolean', 'When', 'Who', 'What', 'Where'},
 'PP-DATE': {'What', 'How', 'When', 'Which'},
 'VP-TIME': {'How', 'When', 'Which', 'What'},
 'NP-LAW': {'Which', 'When', 'Who', 'What', 'Where', 'Other'},
 'ADVP-PERSON': {'What', 'Who'},
 'NP-TIME': {'How', 'Which', 'When', 'What', 'Where'},
 'PP-PERCENT': {'What', 'How', 'When'},
 'PP-FAC': {'What', 'Which'},
 'UCP-UNK': {'What', 'Who', 'Which', 'Where'},
 'ADVP-GPE': {'What', 'Which', 'Where'},
 'PP-LOC': {'What', 'Where'},
 'UCP-DATE': {'How'},
 'ADVP-ORG': {'What', 'Who', 'Which'},
 'PP-LAW': {'What', 'Which'},
 'ADVP-CARDINAL': {'How', 'When'},
 'PP-EVENT': {'What'},
 'UCP-PERSON': {'What'},
 'PP-MONEY': {'How'},
 'PP-QUANTITY': {'How', 'What'},
 'PP-NORP': {'What'}}
{'NP-PERSON': Counter({'Who': 2247, 'What': 967, 'Which': 436, 'Where': 76, 'Other': 17, 'When': 9, 'How': 6, 'Boolean': 1}),
 'NP-UNK': Counter({'What': 17375, 'Who': 2634, 'Which': 1604, 'How': 1180, 'Where': 1056, 'When': 509, 'Why': 271, 'Other': 183, 'Boolean': 142}),
 'NP-FAC': Counter({'What': 351, 'Where': 125, 'Which': 73, 'Who': 24, 'When': 10, 'Other': 8, 'Why': 1}),
 'VP-DATE': Counter({'What': 963, 'When': 597, 'How': 310, 'Which': 96, 'Who': 2, 'Where': 1, 'Other': 1}),
 'VP-UNK': Counter({'What': 15289, 'How': 2221, 'Who': 1199, 'Which': 1029, 'Why': 885, 'Boolean': 791, 'Where': 665, 'When': 612, 'Other': 199}),
 'VP-CARDINAL': Counter({'How': 3011, 'What': 319, 'When': 40, 'Which': 22, 'Where': 5, 'Who': 2, 'Other': 2}),
 'NP-DATE': Counter({'When': 3314, 'What': 2841, 'How': 348, 'Which': 264, 'Where': 12, 'Other': 10, 'Who': 6, 'Boolean': 1}),
 'NP-GPE': Counter({'What': 1486, 'Where': 650, 'Which': 416, 'Who': 266, 'When': 12, 'Other': 10, 'How': 8, 'Boolean': 1, 'Why': 1}),
 'NP-ORG': Counter({'What': 2909, 'Who': 1023, 'Which': 647, 'Where': 326, 'Other': 24, 'When': 22, 'How': 21, 'Boolean': 3, 'Why': 1}),
 'NP-CARDINAL': Counter({'How': 573, 'What': 325, 'When': 57, 'Which': 7, 'Other': 2, 'Who': 2, 'Boolean': 1, 'Where': 1}),
 'VP-ORG': Counter({'What': 1270, 'Who': 307, 'Which': 293, 'Where': 76, 'Other': 15, 'How': 11, 'When': 8, 'Boolean': 3, 'Why': 2}),
 'VP-PERSON': Counter({'Who': 1283, 'What': 683, 'Which': 390, 'Where': 15, 'Other': 9, 'When': 7, 'How': 5, 'Why': 1}),
 'VP-WORK_OF_ART': Counter({'What': 243, 'Which': 30, 'Who': 14, 'How': 4, 'Where': 2, 'Why': 1}),
 'NP-WORK_OF_ART': Counter({'What': 334, 'Which': 53, 'Who': 13, 'Where': 6, 'When': 5, 'How': 4, 'Other': 1}),
 'NP-PERCENT': Counter({'What': 556, 'How': 204, 'Other': 3, 'Which': 2, 'Boolean': 1, 'Who': 1, 'Where': 1, 'When': 1}),
 'NP-QUANTITY': Counter({'How': 130, 'What': 62, 'When': 9, 'Where': 2, 'Which': 1}),
 'VP-ORDINAL': Counter({'What': 33, 'Where': 20, 'Which': 9, 'How': 7, 'When': 3}),
 'PP-ORG': Counter({'What': 11, 'Which': 1, 'Where': 1}),
 'PP-UNK': Counter({'What': 300, 'When': 291, 'How': 274, 'Where': 226, 'Why': 46, 'Who': 22, 'Which': 19, 'Boolean': 17}),
 'VP-PERCENT': Counter({'What': 400, 'How': 120, 'When': 3, 'Which': 1, 'Other': 1}),
 'VP-GPE': Counter({'What': 447, 'Which': 133, 'Where': 125, 'Who': 65, 'Other': 6, 'When': 3, 'How': 1}),
 'NP-MONEY': Counter({'How': 159, 'What': 90, 'When': 3, 'Where': 1}),
 'VP-MONEY': Counter({'How': 165, 'What': 81, 'Where': 1}),
 'VP-QUANTITY': Counter({'How': 204, 'What': 63, 'When': 3, 'Who': 1}),
 'NP-LOC': Counter({'What': 422, 'Where': 132, 'Which': 112, 'Who': 21, 'When': 6, 'Other': 4, 'How': 2}),
 'VP-LOC': Counter({'What': 135, 'Which': 29, 'Where': 24, 'Who': 6, 'How': 2, 'When': 1}),
 'VP-NORP': Counter({'What': 434, 'Which': 78, 'Who': 56, 'Where': 6, 'How': 6, 'Boolean': 1}),
 'NP-NORP': Counter({'What': 451, 'Who': 108, 'Which': 69, 'Where': 6, 'Other': 3, 'How': 3, 'Boolean': 2, 'When': 1}),
 'PP-PERSON': Counter({'Who': 17, 'What': 12, 'Which': 2}),
 'NP-PRODUCT': Counter({'What': 77, 'Who': 20, 'Which': 16, 'Where': 4, 'When': 3, 'Other': 1}),
 'NP-EVENT': Counter({'What': 274, 'Which': 60, 'When': 22, 'Where': 11, 'Who': 4, 'Why': 3, 'Other': 2, 'How': 1}),
 'PP-WORK_OF_ART': Counter({'What': 6, 'Which': 1}),
 'VP-EVENT': Counter({'What': 93, 'Which': 12, 'Where': 4, 'When': 3, 'Who': 3, 'How': 2}),
 'ADVP-WORK_OF_ART': Counter({'Which': 2, 'What': 1}),
 'ADVP-UNK': Counter({'How': 79, 'What': 74, 'Where': 29, 'Which': 18, 'When': 17, 'Boolean': 16, 'Who': 5, 'Other': 1, 'Why': 1}),
 'VP-LAW': Counter({'What': 43, 'Which': 7, 'When': 4, 'Who': 3, 'How': 1, 'Where': 1}),
 'VP-LANGUAGE': Counter({'What': 15, 'Which': 2}),
 'NP-ORDINAL': Counter({'What': 32, 'Where': 21, 'Which': 15, 'How': 6, 'When': 2}),
 'VP-FAC': Counter({'What': 108, 'Which': 28, 'Where': 17, 'Who': 10, 'When': 2, 'Other': 2, 'How': 1}),
 'ADVP-ORDINAL': Counter({'What': 12, 'Where': 10, 'Which': 1}),
 'NP-LANGUAGE': Counter({'What': 44, 'Which': 5, 'Boolean': 1}),
 'PP-GPE': Counter({'What': 4, 'Which': 1, 'Where': 1, 'Who': 1}),
 'ADVP-DATE': Counter({'How': 22, 'When': 17, 'What': 14, 'Which': 1, 'Where': 1}),
 'ADVP-NORP': Counter({'What': 1, 'Which': 1}),
 'PP-CARDINAL': Counter({'How': 18, 'When': 4, 'What': 2}),
 'VP-PRODUCT': Counter({'What': 39, 'Who': 6, 'Which': 4, 'How': 1, 'Boolean': 1, 'When': 1, 'Where': 1}),
 'PP-DATE': Counter({'When': 31, 'What': 25, 'How': 7, 'Which': 2}),
 'VP-TIME': Counter({'How': 42, 'What': 23, 'When': 5, 'Which': 3}),
 'NP-LAW': Counter({'What': 156, 'Which': 26, 'When': 8, 'Where': 8, 'Who': 5, 'Other': 1}),
 'ADVP-PERSON': Counter({'What': 3, 'Who': 1}), 'NP-TIME': Counter({'What': 36, 'How': 28, 'When': 10, 'Where': 1, 'Which': 1}),
 'PP-PERCENT': Counter({'What': 12, 'How': 4, 'When': 1}),
 'PP-FAC': Counter({'What': 3, 'Which': 1}),
 'UCP-UNK': Counter({'What': 20, 'Which': 3, 'Who': 2, 'Where': 1}),
 'ADVP-GPE': Counter({'Which': 2, 'What': 1, 'Where': 1}),
 'PP-LOC': Counter({'What': 4, 'Where': 1}),
 'UCP-DATE': Counter({'How': 1}),
 'ADVP-ORG': Counter({'What': 3, 'Who': 2, 'Which': 1}),
 'PP-LAW': Counter({'What': 1, 'Which': 1}),
 'ADVP-CARDINAL': Counter({'How': 1, 'When': 1}),
 'PP-EVENT': Counter({'What': 3}),
 'UCP-PERSON': Counter({'What': 1}),
 'PP-MONEY': Counter({'How': 3}),
 'PP-QUANTITY': Counter({'What': 3, 'How': 1}),
 'PP-NORP': Counter({'What': 2})}
"""


answertag2qtype_infos = load(ANSWERTAG2QTYPE_FILE_PATH)
ANSWERTAG2QTYPE_SET = answertag2qtype_infos["answertag2qtype_set"]
# refined set: remove low frequency types.
ANSWERTAG2QTYPE_COUNTER = answertag2qtype_infos["answertag2qtype_counter"]
print("Before delete low frequency types: ==============")
for k in ANSWERTAG2QTYPE_SET:
    print(k)
    print(ANSWERTAG2QTYPE_SET[k])

for ans_tag in ANSWERTAG2QTYPE_SET:
    counter = ANSWERTAG2QTYPE_COUNTER[ans_tag]
    threshold = 0.03 * sum(counter.values())
    for q_type in counter:
        if counter[q_type] < threshold:
            ANSWERTAG2QTYPE_SET[ans_tag].remove(q_type)

print("After delete low frequency types: ==============")
for k in ANSWERTAG2QTYPE_SET:
    print(k)
    print(ANSWERTAG2QTYPE_SET[k])


def augment_qg_data(context):
    """
    Given context, get all possible (context, answer, clue, style) tuples.
    """
    result = {"context": context, "selected_infos": []}

    answers = select_answers(context)

    for ans in answers:
        (answer_text, char_st, char_ed, answer_bio_ids, label) = ans
        info = {"answer": {"answer_text": answer_text, "char_start": char_st, "char_end": char_ed,
                           "answer_bio_ids": answer_bio_ids, "answer_chunk_tag": label},
                "styles": None,
                "clues": None}

        styles = select_question_types(context, char_st, char_ed, ANSWERTAG2QTYPE_SET)
        info["styles"] = list(styles)

        clues = select_clues(context, answer_text, answer_bio_ids, max_dependency_distance=6)
        info["clues"] = clues

        result["selected_infos"].append(info)

    return result


if __name__ == "__main__":
    pass

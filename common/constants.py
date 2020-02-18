import os
import ahocorasick
import spacy
from spacy.symbols import ORTH
import sentencepiece as spm
import gensim
import torch
import benepar


print("Start loading constants ...")

# data path
current_path = os.getcwd().split("/")
#current_path = "/Users/bangliu/Documents/Work/CurrentWork/FQG/src/model/FactorizedQG".split("/")

DATA_PATH = "/".join(current_path[:-4]) + "/Datasets/"
PROJECT_PATH = "/".join(current_path[:-4]) + "/FQG/"

CODE_PATH = PROJECT_PATH + "src/model/FactorizedQG/"
OUTPUT_PATH = PROJECT_PATH + "output/"
CHECKPOINT_PATH = PROJECT_PATH + "output/checkpoint/"
FIGURE_PATH = PROJECT_PATH + "output/figure/"
LOG_PATH = PROJECT_PATH + "output/log/"
PKL_PATH = PROJECT_PATH + "output/pkl/"
RESULT_PATH = PROJECT_PATH + "output/result/"


FUNCTION_WORDS_FILE_PATH = DATA_PATH + "original/function-words/function_words.txt"
FIXED_EXPRESSIONS_FILE_PATH = DATA_PATH + "original/fixed-expressions/fixed_expressions.txt"

BPE_MODEL_PATH = DATA_PATH + "original/BPE/en.wiki.bpe.op50000.model"
BPE_EMB_PATH = DATA_PATH + "original/BPE/en.wiki.bpe.op50000.d100.w2v.txt"

GLOVE_BIN_PATH = DATA_PATH + "original/Glove/glove.840B.300d.bin"
GLOVE_TXT_PATH = DATA_PATH + "original/Glove/glove.840B.300d.txt"

# question type
QUESTION_TYPES = [
    "Who", "Where", "When", "Why", "Which", "What", "How",
    "Boolean", "Other"]
INFO_QUESTION_TYPES = [
    "Who", "Where", "When", "Why", "Which", "What", "How"]
BOOL_QUESTION_TYPES = [
    "Am", "Is", "Was", "Were", "Are",
    "Does", "Do", "Did",
    "Have", "Had", "Has",
    "Could", "Can",
    "Shall", "Should",
    "Will", "Would",
    "May", "Might"]
Q_TYPE2ID_DICT = {
    "What": 0, "Who": 1, "How": 2,
    "Where": 3, "When": 4, "Why": 5,
    "Which": 6, "Boolean": 7, "Other": 8}

# function words
f_func_words = open(FUNCTION_WORDS_FILE_PATH, "r", encoding="utf-8")
func_words = f_func_words.readlines()

FUNCTION_WORDS_LIST = [word.rstrip() for word in func_words]


# fixed expressions
f_fixed_expression = open(FIXED_EXPRESSIONS_FILE_PATH, "r", encoding="utf-8")
fixed_expressions = f_fixed_expression.readlines()

FIXED_EXPRESSIONS_LIST = [word.rstrip() for word in fixed_expressions]


# AC Automaton
AC_AUTOMATON = ahocorasick.Automaton()
for idx, key in enumerate(FIXED_EXPRESSIONS_LIST):
    AC_AUTOMATON.add_word(key, (idx, key))
AC_AUTOMATON.make_automaton()


# BPE
SPM = spm.SentencePieceProcessor()
SPM.Load(BPE_MODEL_PATH)

# special tokens
SPECIAL_TOKENS = {"pad": "<pad>", "oov": "<oov>", "sos": "<sos>", "eos": "<eos>"}
SPECIAL_TOKEN2ID = {"<pad>": 0, "<oov>": 1, "<sos>": 2, "<eos>": 3}

# spaCy
NLP = spacy.load("en")
# prevent tokenizer split special tokens
for special_token in SPECIAL_TOKENS.values():
    NLP.tokenizer.add_special_case(special_token, [{ORTH: special_token}])

# benepar
PARSER = benepar.Parser("benepar_en2")

# glove
GLOVE = gensim.models.KeyedVectors.load_word2vec_format(GLOVE_BIN_PATH, binary=True)

# device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# env
EXP_PLATFORM = "others"  # set it to be "venus" or any other string. This is just used for run experiments on Venus platform.

print("Finished loading constants ...")

import unittest
from FQG_data_utils import *
from config import *
from common.constants import FUNCTION_WORDS_LIST


class TestDataUtils(unittest.TestCase):

    def test_get_related_words(self):
        print("\nSTART: test_get_related_words")
        print(get_related_words("big", 10))
        print(get_related_words("Small", 10))
        print(get_related_words("small", 10))
        print(get_related_words("SmAlL", 10))
        print(get_related_words("xxyyabc", 10))
        print("END: test_get_related_words\n")

    def test_get_related_words_dict(self):
        print("\nSTART: test_get_related_words_dict")
        print(get_related_words_dict(["how", "a", "big", "small", "Happy"], 10))
        print("END: test_get_related_words_dict\n")

    def test_get_related_words_ids_mat(self):
        print("\nSTART: test_get_related_words_ids_mat")
        print(get_related_words_ids_mat(
            {"<pad>": 0, "<oov>": 1, "<sos>": 2, "<eos>": 3, "how": 4, "a": 5, "big": 6, "small": 7}, 10))
        print("END: test_get_related_words_ids_mat\n")

    def test_get_related_words_ids_mat_with_related_words_dict(self):
        print("\nSTART: get_related_words_ids_mat_with_related_words_dict")
        print(get_related_words_ids_mat_with_related_words_dict(
            word2id_dict={"<pad>": 0, "<oov>": 1, "<sos>": 2, "<eos>": 3, "how": 4, "a": 5, "big": 6, "small": 7},
            topN=3,
            related_words_dict={"<pad>": {"semantic_related": ["<pad>"]},
                                "<oov>": {"semantic_related": ["<oov>"]},
                                "<sos>": {"semantic_related": ["<sos>"]},
                                "<eos>": {"semantic_related": ["<eos>"]},
                                "how": {"semantic_related": ["a", "big", "small"]},
                                "a": {"semantic_related": ["how", "big", "small"]},
                                "big": {"semantic_related": ["how", "a", "small"]},
                                "small": {"semantic_related": ["how", "a", "big"]}}))
        print("END: get_related_words_ids_mat_with_related_words_dict\n")

    def test_get_softcopy_ids(self):
        input = "What is wrong with you"
        output = "What are correct and you, incorrect right correctly"
        sent_length = 10
        topN = 20
        result = get_softcopy_ids(input, output, sent_length, topN)
        expected_result = [1, 2, 4, 5, 1, 0, 3, 4, 4, 0]
        self.assertTrue((result == expected_result).all())

    def test_get_copy_labels(self):
        print("\nSTART: get_copy_labels")
        print(get_copy_labels(
            input_tokens="Tom is 8 years old . He likes playing football .".split(),
            output_tokens="How old is Tom ? What does he like ?".split(),
            input_padded_length=15,
            output_padded_length=15,
            word2id_dict={"tom": 0, "is": 1, "8": 2, "years": 3, "old": 4,
                          ".": 5, "he": 6, "likes": 7, "playing": 8, "football": 9,
                          "how": 10, "?": 11, "what": 12, "does": 13, "like": 14,
                          "<pad>": 100, "<oov>": 101, "<sos>": 102, "<eos>": 103},
            related_word_ids_mat=[[6],
                                  [12, 13],
                                  [],
                                  [4],
                                  [3],
                                  [11],
                                  [0, 1],
                                  [14],
                                  [7, 9],
                                  [7, 8],
                                  [12],
                                  [5],
                                  [10],
                                  [1],
                                  [7]]))
        print("END: get_copy_labels\n")

    def test_get_clue_ids(self):
        result = get_clue_ids(
            "Tom is 8 years old. He likes playing football.",
            np.array([1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]),
            "How old is Tom? What does he like?",
            15, 20, None)
        expected_result = [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        print(result)
        self.assertTrue((result == expected_result).all())

    def test_get_clue_ids_with_input_copied_hard_soft(self):
        print("\nSTART: get_clue_ids_with_input_copied_hard_soft")
        print(get_clue_ids_with_input_copied_hard_soft(
            input_copied_hard_soft_padded=np.array([2, 3, 4, 5, 6, 7, 8, 9, 3]),
            context_is_content_ids_padded=np.array([0, 1, 1, 1, 0, 1, 0, 1, 1]),
            topN=5))
        print("END: get_clue_ids_with_input_copied_hard_soft\n")

    def test_get_content_ids(self):
        sentence = "Mary has lived in England for ten years."
        result = get_content_ids(sentence, FUNCTION_WORDS_LIST, 15)
        expected_result = [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        self.assertTrue((result == expected_result).all())

    def test_get_question_type(self):
        question = "How are you?"
        result = get_question_type(question)
        expected_result = ('How', 2)
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()

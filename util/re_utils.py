# -*- coding: utf-8 -*-
"""
Util functions related to regular expression.
"""
import re


def get_match_spans(pattern, input):
    """
    Given string pattern and string input,
    return list of [) char position tuples of patterns in input.
    :param pattern: string pattern to match.
    :param input: string input where we find pattern.
    :return: a list of pattern char position tuples in input.
    """
    spans = []
    for match in re.finditer(re.escape(pattern), input):
        spans.append(match.span())
    return spans


if __name__ == "__main__":
    # test get_match_spans
    pattern = "are"
    input = "how are you are you"
    print(get_match_spans(pattern, input))
    # we will get [(4, 7), (12, 15)]

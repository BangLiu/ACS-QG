# -*- coding: utf-8 -*-
from collections import Counter, OrderedDict
from scipy.linalg import norm


def sort_dict_by_key_str(d):
    """
    Sort dictionary by its key values.
    """
    return OrderedDict(
        sorted(d.items(), key=lambda t: t[0]))


def cosine_sim(a, b):
    if len(b) < len(a):
        a, b = b, a
    res = 0
    for key, a_value in a.iteritems():
        res += a_value * b.get(key, 0)
    if res == 0:
        return 0
    try:
        res = res / norm(a.values()) / norm(b.values())
    except ZeroDivisionError:
        res = 0
    return res


def counter2ordered_dict(counter, N=None):
    if N is None:
        return OrderedDict(counter.most_common())
    else:
        assert N > 0
        return OrderedDict(counter.most_common(N))


def get_value_maps_between_dicts(dict_a, dict_b):
    dict_a2dict_b = {}
    for k in dict_a:
        dict_a2dict_b[dict_a[k]] = dict_b.get(k, dict_b["<OOV>"])
    return dict_a2dict_b


if __name__ == "__main__":
    # test counter2ordered_dict
    x = Counter({'a': 5, 'b': 3, 'c': 7})
    d = counter2ordered_dict(x)
    print(d)
    d = counter2ordered_dict(x, 2)
    print(d)
    for k in d:
        print(k, d[k])
    for k, v in d.items():
        print(k, v)

    # test get_value_maps_between_dicts
    dict_a = {"w1": 1, "w2": 2, "w3": 3, "w4": 4, "OOV": 0}
    dict_b = {"w1": 11, "w2": 22, "w5": 5, "w4": 4, "OOV": 0}
    print(get_value_maps_between_dicts(dict_a, dict_b))

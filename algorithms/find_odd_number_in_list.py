from operator import xor
from functools import reduce


def find_odd_repeating_in_array(array) -> float:
    """
    param array: array contaiining element which repeats odd number of times

    return: element repeating odd number of times
    """

    return reduce(lambda x, y: xor(x, y), array)


def _check(array):
    return reduce(lambda number, tuple_from_counter: number + tuple_from_counter[0] * (tuple_from_counter[1] % 2),
                  Counter(array).items(), 0)


if __name__ == '__main__':
    from collections import Counter
    assert find_odd_repeating_in_array([1, 1, 1, 2, 2]) == _check([1, 1, 1, 2, 2])

"""
Provides function for finding editing remarks
"""


def levenstein1(a, b) -> int:
    """
    :param a: string to transform
    :param b: string to compare with
    :return: length of minimum editing distance required for transforming a into b
    >>> levenstein1('distance', 'editing')
    5

    """

    D = [[i + j if i * j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            D[i][j] = min({D[i-1][j] + 1, D[i][j-1] + 1, D[i-1][j-1] + (a[i-1] != b[j-1])})
    return D[-1][-1]


def levenstein2(a, b) -> int:
    """
    O(min(len(a),len(b)) memory
    :param a: string to compare
    :param b: string as matrix
    :return:

    >>> levenstein2('distance', 'editing')
    """
    if len(a) < len(b):
        return levenstein2(b, a)


if __name__ == '__main__':
    import doctest
    doctest.testmod()

def _get_ceil_index(arr, T, l, r, key):
    while r - l > 1:
        m = l + (r - l) // 2
        if arr[T[m]] >= key:
            r = m
        else:
            l = m
    return r


def longest_common_subsuquence(A, B) -> int:
    """

    :param A: 1st fragment
    :param B: 2nd fragment
    :return: length of the longest common subsequence
    """
    m = len(A)
    n = len(B)
    F = [[0] * (n + 1) for _ in range(m + 1)] # Крайние случаи уже подсчитаны
    for i in range(1, m + 1):
        for j in range(1, n+1):
            if A[i - 1] == B[j - 1]:
                F[i][j] = 1 + F[i - 1][j - 1]
            else:
                F[i][j] = max(F[i - 1][j], F[i][j - 1])
    return F[-1][-1]


def lis(A):
    """

    :param A: SEQUENCE
    """
    tailIndices = [0 for _ in range(len(A) + 1)]
    prevIndices = [-1 for _ in range(len(A) + 1)]

    length = 1
    for i in range(1, len(A)):
        if A[i] < A[tailIndices[0]]:
            tailIndices[0] = i
        elif A[i] > A[tailIndices[length - 1]]:
            prevIndices[i] = tailIndices[length - 1]
            tailIndices[length] = i
            length += 1
        else:
            pos = _get_ceil_index(A, tailIndices, -1, length - 1, A[i])
            prevIndices[i] = tailIndices[pos - 1]
            tailIndices[pos] = i

    # sample = ', '.join(map(str, A[:5]))
    # print(f'LIS of given input {sample}... (only first 5 members shown)')
    i = tailIndices[length - 1]
    L = [None for _ in range(length)]
    for j in range(length - 1, -1, -1):
        L[j] = A[i]
        i = prevIndices[i]
    yield length
    yield L


def longest_folding_subseq(a):
    d = [0 for _ in range(len(a))]
    for i in range(len(a)):

        d[i] = 1
        for j in range(i):
            if a[i] % a[j] == 0 and d[j] + 1 > d[i]:
                d[i] = d[j] + 1
    ans = 0
    for i in range(len(a)):
        ans = max(ans, d[i])
    return ans


def longest_nondecreasing_subseq(a):
    TailInd = [0 for _ in range(len(a) + 1)]
    PrevInd = [-1 for _ in range(len(a) + 1)]
    length = 1
    for i in range(1, len(a)):
        if a[i] < a[TailInd[0]]:
            TailInd[0] = 1
        elif a[i] >= a[TailInd[length - 1]]:
            PrevInd[i] = TailInd[length - 1]
            TailInd[length] = i
            length += 1
        else:
            pos = _get_ceil_index(a, TailInd, -1, length - 1, a[i])
            PrevInd[i] = TailInd[pos - 1]
            TailInd[pos] = i

    i = TailInd[length - 1]
    L = [None for _ in range(length)]
    for j in range(length - 1, -1, -1):
        L[j] = a[i]
        i = PrevInd[i]
    yield length
    yield L


def longest_non_increasing_subseq(a):
    """
    :param a: sequence
    :return: number and sequence
    """
    def _getInd(arr, T, l, r, key):
        while r - l > 1:
            m = l + (r - l) // 2
            if arr[T[m]] < key:
                r = m
            else:
                l = m
        return r

    TailInd = [0 for _ in range(len(a) + 1)]
    PrevInd = [-1 for _ in range(len(a) + 1)]

    length = 1
    for i in range(1, len(a)):
        if a[i] > a[TailInd[0]]:
            TailInd[0] = i
        elif a[i] <= a[TailInd[length - 1]]:
            PrevInd[i] = TailInd[length - 1]
            TailInd[length] = i
            length += 1
        else:
            pos = _getInd(a, TailInd, -1, length-1, a[i])
            PrevInd[i] = TailInd[pos-1]
            TailInd[pos] = i
    i = TailInd[length - 1]
    L = [None for _ in range(length)]
    for j in range(length-1, -1, -1):
        L[j] = a[i]
        i = PrevInd[i]
    yield length
    yield L


lnds = longest_nondecreasing_subseq
lnis = longest_non_increasing_subseq


def lnis_indeces(a):
    """
    >>> lnis_indeces([32, 27, 74, 20, 27, 34, 7, 41, 65, 66, 19, 75, 58, 38, 49,85, 4, 50])
    5
    3 10 13 15 17
    """
    def _getInd(arr, T, l, r, key):
        while r - l > 1:
            m = l + (r - l) // 2
            if arr[T[m]] < key:
                r = m
            else:
                l = m
        return r

    TailInd = [0 for _ in range(len(a) + 1)]
    PrevInd = [-1 for _ in range(len(a) + 1)]

    length = 1
    for i in range(1, len(a)):
        if a[i] > a[TailInd[0]]:
            TailInd[0] = i
        elif a[i] <= a[TailInd[length - 1]]:
            PrevInd[i] = TailInd[length - 1]
            TailInd[length] = i
            length += 1
        else:
            pos = _getInd(a, TailInd, -1, length - 1, a[i])
            PrevInd[i] = TailInd[pos - 1]
            TailInd[pos] = i
    print(length)

    i = TailInd[length - 1]
    indeces = [None for _ in range(length)]
    for j in range(length-1, -1, -1):
        indeces[j] = i + 1
        i = PrevInd[i]
    print(*indeces)


if __name__ == '__main__':
    import doctest
    doctest.testmod()

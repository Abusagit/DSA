from collections import defaultdict


def prefix(s: str) -> list:
    """
    Creates prefix function for KMP algorithm
    """
    P = [0 for _ in range(len(s))]
    i, j = 0, 1
    while j < len(s):
        if s[i] == s[j]:
            P[j] = i + 1
            i += 1
            j += 1
        # s[i] != s[j]
        elif i:  # i > 0
            i = P[i - 1]
        else:
            P[j] = 0
            j += 1
    return P


def kmp(string, substring):
    """
    Performs KMP search
    """
    sub_len = len(substring)
    str_len = len(string)
    if not str_len or sub_len > str_len:
        return None
    P = prefix(substring)
    print(f'>>> {P}')
    entries = []
    i = j = 0
    while i < str_len and j < sub_len:
        if string[i] == substring[j]:
            if j == sub_len - 1:
                entries.append(i - sub_len + 1)
                j = 0
            else:
                j += 1
            i += 1
        # string[i] != substring[j]:
        elif j:  # j > 0
            j = P[j - 1]
        else:
            i += 1
    return entries


def kmp_with_crossed_entries(string, substring, symbol='@'):
    """
    Performs KMP search with crossing entries
    """
    entries = []
    mega = substring + symbol + string
    length = len(mega)
    P = [0 for _ in range(length)]
    i, j = 0, 1
    while j < length:
        if mega[i] == mega[j]:
            P[j] = i + 1
            if P[j] == len(substring):
                entries.append(j - 2 * len(substring))
            j += 1
            i += 1
        elif i:
            i = P[i - 1]
        else:
            P[j] = 0
            j += 1
    return entries


def turbo_bmh(string, pattern):
    def _suffices_preprocessing(suffix):
        suffix[m - 1] = m
        g = m - 1

        for i in range(m - 2, -1, -1):
            if i > g and suffix[i + m - f - 1] < i - g:
                suffix[i] = suffix[i + m - 1 - f]
            else:
                if i < g:
                    g = i
                f = i
                while g >= 0 and pattern[g] == pattern[g + m - 1 - f]:
                    g -= 1
                suffix[i] = f - g

    def good_suffix_preprocessing():
        # nonlocal pattern, good_suf, m_
        suffix = [0 for _ in range(m)]
        _suffices_preprocessing(suffix)
        for i in range(m):
            good_suf[i] = m
        for i in range(m - 1, -1, -1):
            if suffix[i] == i + 1:
                for j in range(m - 1 - i):
                    if good_suf[j] == m:
                        good_suf[j] = m - 1 - i

        for i in range(0, m - 1):
            good_suf[m - 1 - suffix[i]] = m - 1 - i

    n = len(string)
    m = len(pattern)
    good_suf = [0 for i in range(m)]
    bad_character = [m for _ in range(256)]
    for k in range(m - 1):
        bad_character[ord(pattern[k])] = m - k - 1
    bad_character = tuple(bad_character)
    answers = []

    good_suffix_preprocessing()

    j = 0
    while j <= n - m:
        i = m - 1
        while i >= 0 and pattern[i] == string[i + j]:
            i -= 1

        if i < 0:
            answers.append(j)
            j += good_suf[0]
        else:
            j += max(good_suf[i], bad_character[ord(string[i + j])] - m + 1 + i)

    return answers


def _bmh_bad_character(string, pattern):
    """
    Bad character heuristic
    """
    m = len(pattern)
    n = len(string)

    if m > n:
        return -1

    skip = [m for _ in range(256)]
    answer = []
    for k in range(m - 1):
        skip[ord(pattern[k])] = m - k - 1
    skip = tuple(skip)
    k = m - 1

    while k < n:
        j = m - 1
        i = k
        while j >= 0 and string[i] == pattern[j]:
            j -= 1
            i -= 1
        if j == -1:
            answer.append(i + 1)
        k += skip[ord(string[k])]
    return answer


def rabin_karp(text, pattern):
    answer = []
    n = len(text)
    m = len(pattern)
    q = 1000000007
    p = t = 0
    h = 1
    d = 10
    for i in range(m - 1):
        h = (h * d) % q
    # Calculate hash value for pattern and text
    for i in range(m):
        p = (d * p + ord(pattern[i])) % q
        t = (d * t + ord(text[i])) % q

    # Find the match
    for i in range(n - m + 1):
        if p == t:
            for j in range(m):  # this symbol-by-symbol compare slows algorithm a little bit and it can go without it
                if text[i + j] != pattern[j]:
                    break
            else:
                answer.append(i)
        if i < n - m:
            t = (d * (t - ord(text[i]) * h) + ord(text[i + m])) % q

            if t < 0:
                t += q
    return answer

if __name__ == '__main__':
    g = '1'
    f = 'asdadsasd'
    print(kmp(f, g))
    # print(prefix('AABAA'))
    s = 'aabaabaaaabaabaaab'
    sub = 'aabaa'
    print(kmp(s, sub))
    print(kmp_with_crossed_entries(s, sub))
    print(_bmh_bad_character(s, sub))
    a = 'GCATCGCAGAGAGTATACAGTACG'
    b = 'GCAGAGAG'
    print(turbo_bmh(s, sub))
    tests = [
        [[8, 25], 'the', 'this is the string to do the search in'],
        [[0, 2, 10], 'co', 'cocochanelco'],
        [[2, 6], 'co', 'mycocacola'],
        [[2, 4, 6, 9], 'co', 'mycococoacola'],
        [[2, 4], 'coco', 'mycococoacola'],
        [[10], 'co', 'lalalalalaco'],
        [[0], 'co', 'colalalalala'],
        [[], 'a', 'zzzzzzzzzzz'],
        [[0], 'a', 'a'],
        [[], 'z', 'a'],
        [[], 'z', 'aa'],
        [[1], 'z', 'az'],
        [[0], 'z', 'za'],
        [[r for r in range(11)], 'z', 'zzzzzzzzzzz'],
        [[5, 6], 'z', 'aaaaazzaaaaa'],
    ]
    print()
    for test in tests:
        print(turbo_bmh(test[2], test[1]))
        assert turbo_bmh(test[2], test[1]) == test[0]
        assert rabin_karp(test[2], test[1]) == test[0]

    print(turbo_bmh(s, sub))
    print(rabin_karp(s, sub))

import operator as op


def reverse_polish_notation(*pargs):
    """
    for multiplication, divition, summation, subtraction

    [5, 2, '+'] <-> 5 + 2
    [2, 7, '+', 5, '*'] <-> (2 + 7) * 5
    :return: result
    """
    stack = []
    operator = {'+': op.add,
                '-': op.sub,
                '*': op.mul,
                '/': op.truediv,
                '//': op.floordiv}
    for token in pargs:
        if isinstance(token, (float, int)):
            stack.append(token)
        else:
            y = stack.pop()
            x = stack.pop()
            z = operator[token](x, y)
            stack.append(z)
    return stack.pop()


def brackets_check(s) -> str or int:
    """:key
    ckecks brackets sequence correction
    >>> brackets_check('((()))[][()]')
    'Success'
    >>> brackets_check('()')
    'Success'
    >>> brackets_check('(([()))')
    6
    >>> brackets_check('(')
    1
    >>> brackets_check('{{[()]]')
    7
    >>> brackets_check('{}')
    'Success'
    """
    stack = []
    i = 1
    for letter in s:
        if letter not in '()[]{}':
            i += 1
            continue
        if letter in '([{':
            stack.append((letter, i))
        else:
            assert letter in ')]}', f'closing bracket expected, got {letter}'
            if not stack:
                return i
            left, index = stack.pop()
            if left == '(':
                right = ')'
            elif left == '[':
                right = ']'
            else:
                right = '}'
            if right != letter:
                return i
        i += 1
    return 'Success' if not stack else stack.pop()[1]


if __name__ == '__main__':
    a = reverse_polish_notation(2, 3, '-', -5, '*')
    print(a, end=f'\n{"-" * 50}\n')
    import doctest
    doctest.testmod(verbose=True)
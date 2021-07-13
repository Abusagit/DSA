def generate_numbers(N:int=2, M:int=3, prefix:list=None)-> list:
    '''
    Generating numbers with leading zeros
    with combinating all possible conditions for every position
    for N-numerical system M-1-sigits contained number

    N: numeral system base
    M: number of digits in number to generate
    prefix: already permutated part of number
    '''

    if M == 0:
        print(*prefix)
        return
    prefix = prefix or []
    for digit in range(N):
        prefix.append(digit)
        generate_numbers(N, M-1, prefix)
        prefix.pop()

def binom_coeff_recur(n, k):
    if n == k or k == 0:
        return 1
    return binom_coeff_recur(n - 1, k - 1) + binom_coeff_recur(n - 1, k)


def binom_coeff_fact(n, k):
    numerator, denominator = 1, 1
    for i in range(2, n + 1):
        numerator *= i
    for j in range(2, k + 1):
        denominator *= j
    for j in range(2, (n - k) + 1):
        denominator *= j

    return int(numerator / denominator)


if __name__ == '__main__':
    for i in (binom_coeff_recur, binom_coeff_fact):
        print(f'{i.__name__}: {i(3, 2)}')
    generate_numbers(4, 3)
    print(generate_numbers.__doc__)
    print(generate_numbers.__annotations__)
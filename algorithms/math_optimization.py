def fastpow(a: float, n: int):
    if n == 0:
        return 1
    elif n % 2 == 1:
        return a * fastpow(a, n - 1)
    else:
        return fastpow(a ** 2, n // 2)

def gcd(a, b):
    return a if b == 0 else gcd(b, a % b )


def fib(n):
    f1, f2 = 0, 1
    for _ in range(n - 1):
        f1, f2 = f2, f1 + f2
    return f2


def primes(n):
    """РЕШЕТО ЭРАТОСФЕНА:"""
    a = [i for i in range(n + 1)]
    a[1] = 0
    i = 2
    while i <= n:
        if a[i] != 0:
            for j in range(i, n + 1, i):
                a[j] = 0
            yield i
        i += 1


if __name__ == '__main__':
    print(fpow(3, 7))
    print(fpow(2, 4))

    print(fib(2))
    print(fib(21))
    print(fib(4))
    print(fib(25))

    for j in primes(31):
        print(j)

    print(gcd(5, 10))
    print(gcd(36288, 7875000000))
# fibonacci module

from math import sqrt

def memorize(f):
    cache = {}
    def helper(x):
        if x not in cache:
            cache[x] = f(x)
        return cache[x]
    return helper

@memorize
def fib_recursive(n):
    """
    calculate the nth Fibonacci number exactly with memorization
    O(1.68^n)
    """
    fib = fib_recursive
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)

@memorize
def fib_recursive_faster(n):
    """
    calculate the nth Fibonacci number exactly with memorization
    This function is faster than fib_recursive
    """
    fib = fib_recursive_faster
    if n == 0:
        return 0
    elif n == 1:
        return 1
    elif n % 2 == 0:
        n = n / 2
        return fib(n) * (fib(n) + 2 * fib(n-1))
    else:
        n = (n-1) / 2
        return fib(n+1) ** 2 + fib(n) ** 2

def fib_linear(n):
    """
    calculate the nth Fibonacci number exactly without memorization
    """
    a, b = 0, 1
    for i in range(n):
        a, b = b, a + b
    return a

def fib_matrix(n):
    """calculate the nth Fibonacci number exactly without memorization"""
    def mul(A, B):
        a, b, c = A
        d, e, f = B
        return a * d + b * e, a * e + b * f, b * e + c * f

    def pow(A, n):
        if n == 1:
            return A
        if n & 1 == 0:
            return pow(mul(A, A), n // 2)
        else:
            return mul(A, pow(mul(A, A), (n-1)//2))

    if n < 2:
        return n
    return pow((1, 1, 0), n-1)[0]

def fib_formula(n):
    """calculate the nth Fibonacci number approximately"""
    phi = (1 + sqrt(5)) / 2
    return int(round(phi ** n / sqrt(5) + 1 / 2))

if __name__ == '__main__':
    n = 100
    print fib_recursive(n)
    print fib_recursive_faster(n)
    print fib_linear(n)
    print fib_matrix(n)
    print fib_formula(n)

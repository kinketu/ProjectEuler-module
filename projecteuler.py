# ProjectEuler module

__author__ = "kinketu"
__date__ = "2013/08/31"
__version__ = "0.1"

from math import sqrt, floor, log10
from fractions import Fraction, gcd
from itertools import chain, count
import time
from decimal import Decimal

def lcm(numbers): 
    """
    This function is to return the LCM. 
    This function contains recurrence algorithm. 
    So maybe slower the other algorithm.
    """
    if len(numbers) == 2: 
        num0 = numbers[0] 
        num1 = numbers[1] 
        return num0 * num1 / gcd(num0, num1) 
    else: 
        for i in range(len(numbers)): 
            return lcm([numbers[0], lcm(numbers[1:])]) 

def fib_recursive(n):
    """
    calculate the nth Fibonacci number exactly without memorization
    O(1.68^n)
    """
    fib1 = fib_recursive
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib1(n-1) + fib1(n-2)

def fib_recursive_faster(n):
    """
    calculate the nth Fibonacci number exactly without memorization
    This function is faster than fib_recursive
    """
    fib2 = fib_recursive_faster
    if n == 0:
        return 0
    elif n == 1:
        return 1
    elif n % 2 == 0:
        n = n / 2
        return fib2(n) * (fib2(n) + 2 * fib2(n-1))
    else:
        n = (n-1) / 2
        return fib2(n+1) ** 2 + fib2(n) ** 2

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

def num_reverse(num):
    """
    return the reversed number of the integer
    """
    return int(str(num)[::-1])

def prime_factorization(n):
    """If n is 28, return [2, 2, 7]
    prime_factorization function"""
    result = []
    for i in range(2, n+1):
        s = 0;
        while n / float(i) == floor(n/float(i)):
            n = n / float(i)
            s += 1
        if s > 0:
            for k in range(s):
                result.append(i)
            if n == 1:
                return result

def divisors(n):
    """Return a list of divisors."""
    numbers = []
    for i in range(1, n+1):
        if n % i == 0:
            numbers.append(i)
    return numbers

def proper_divisors(n):
    """Return a list of proper divisors."""
    numbers = []
    for i in range(1, n):
        if n % i == 0:
            numbers.append(i)
            
    return numbers

def count_divisors(n):
    """
    Count the number of divisors
    >>>count_divisors(28)
    >>>6
    the divisors of 28 are 1, 2, 4, 7, 14, 28
    """
    if n == 1:
        return 0
    m = int(sqrt(n))
    c = 1
    if m*m == n:
        c += 1
        m -= 1
    for i in range(2, m+1):
        if n%i == 0:
            c += 2
    return c

def count_proper_divisors(n):
    """count n's number of proper divisor numbers
    If n is 28, return 5 (1, 2, 4, 7, 14).
    """
    if n == 1:
        return 0
    m = int(sqrt(n))
    c = 1
    if m*m == n:
        c += 1
        m -= 1
    for i in range(2, m+1):
        if n%i == 0:
            c += 2
    return c

"""
def count_proper_divisors2(n):
    # todo: time this function's execute time
    return len(proper_divisors(n))
"""

def sum_divisors(n):
    return sum(proper_divisors(n)) + n

def sum_proper_divisors(n):
    return sum(proper_divisors(n))

def is_perfect(n):
    if sum_proper_divisors(n) == n:
        return True
    else:
        return False

def is_deficient(n):
    if sum_proper_divisors(n) < n:
        return True
    else:
        return False

def is_abundant(n):
    if sum_proper_divisors(n) > n:
        return True
    else:
        return False

def alpha_number(alpha):
    """Return a number of alphabet."""
    if alpha.isupper() == False:
        num = ord(alpha) - 96
        return num
    elif alpha.isupper() == True:
        num = ord(alpha) - 64
        return num

def reduce_triangle(to_reduce):
    """
    Reduce 'to_reduce' in place by rolling up the maximum path info one row.

    >>> test = [[3,], \
            [7, 5], \
            [2, 4, 6], \
            [8, 5, 9, 3]]
    >>> test = reduce_triangle(test)
    >>> test
    [[3], [7, 5], [10, 13, 15]]
    >>> test = reduce_triangle(test)
    >>> test
    [[3], [20, 20]]
    >>> test = reduce_triangle(test)
    >>> test
    [[23]]
    """
    last_row = to_reduce[-1]
    for index in xrange(len(to_reduce)-1):
        to_reduce[-2][index] += max(last_row[index:index+2])
    del to_reduce[-1]
    return to_reduce

def permutations(L):
    """Generate and return a list of permutations alphabetically.
    This program is very slow.
    This program is slower than itertools.permutations.
    Especially, length of the list L is over than 8."""
    if L == []:
        return [[]]
    else:
        return [[h]+t for i,h in enumerate(L)
                      for t   in permutations(L[:i]+L[i+1:])]

def num_split(num):
    """Return a list of split numbers.
    There are other ways to split number.
    get_digits function?
    >>>list(str(123))
    >>>["1", "2", "3"]"""
    num = list(str(num))
    return [int(i) for i in num]

def cont_frac(num, index):
    """There are some bags in this function because of truncation error.

    example:
    >>>cont_frac(sqrt(2), 100)
    >[1, 2, 2, ..., 1, 1, ..., 1809, ...]
    """
    b = []
    b.append(int(num))
    for i in xrange(1, index):
        num = 1 / (num - b[i-1])
        #print num
        b.append(int(num))
    return b

def cntfrac2float(fractions):
    """Calculate continued fraction and return a float type number."""
    f = 0
    n = len(fractions)-1
    while n > 0:
        f = 1.0 / (fractions[n] + f)
        #print f
        n -= 1
    return f + fractions[0]

def cntfrac2frac(fractions):
    """Calculate continued fraction and return a fraction."""
    fractions = [Fraction(i) for i in fractions]
    f = Fraction(0)
    n = len(fractions)-1
    while n > 0:
        f = Fraction(Fraction(1) / (fractions[n] + f))
        #print f
        #print type(f)
        n -= 1
    return Fraction(f + fractions[0])

def num_digits(num):
    """Return the number of digits.
    If num = 0, raise Num Domain Error."""
    if num == 0:
        return 1
    return int(log10(num)+1)

def create_corners(side_length):
    return sequence(side_length)

def sequence(side_length):
    """Return a list of numbers
    cf. problem 28, 58"""
    index = side_length
    numbers = []
    tmp1 = (index -1 ) / 2
    #numbers.append([index, 3, 5, 7, 9])
    for i in range(tmp1):
        if i == 0:
            numbers.append([3, 3, 5, 7, 9])
        else:
            diff = (3+i*2) - 1
            tmp2 = numbers[i-1][4] + diff
            numbers.append([3+i*2, tmp2, tmp2+diff, tmp2+diff*2, tmp2+diff*3])
    return numbers

def flatten(nested_list):
    return list(chain.from_iterable(nested_list))

def is_pandigital(n, s=9): n = str(n); return len(n) == s \
                               and not "1234567890"[:s].strip(n)

def is_perm(a, b):
    return sorted(str(a)) == sorted(str(b))

def gen_pandigitals(digit):
    """This function is very slow. expecially digit is over 5.
    If digit is 9, this function return Memmory Error."""
    pandigitals = []
    below = 10 ** (digit-1)
    above = 10 * below
    for i in range(below, above):
        if is_pandigital(i, s=digit):
            pandigitals.append(i)
    return pandigitals

def is_palindrome(string):
    r_string = str_reverse(string)
    cnt = 0
    while cnt < len(string):
        if string[cnt] == r_string[cnt]:
            cnt += 1
            continue
        else:
            return False
        #cnt += 1
    return True

def time_func(func):
    start = time.clock()
    func()
    elapsed = time.clock() - start
    print elapsed, "sec"

def totient(x):
    t = x
    if x % 2 == 0:
        t /= 2
        x /= 2
        while x % 2 == 0:
            x /= 2
    d = 3
    while x / d >= d:
        if x % d == 0:
            t = t / d * (d - 1)
            x /= d
            while x % d == 0:
                x /= d
        d += 2
    if x > 1:
        t = t / x * (x - 1)
    return t

def sub_partition(n, k):
    if n < 0:
        return 0
    if n <= 1 or k == 1:
        return 1
    s = 0
    for i in range(1, k+1):
        s += sub_partition(n-i, i)
    return s

def partition_slow(n):
    """This function is very slow If n is 100."""
    return sub_partition(n, n)

def partition(target):
    ns = xrange(1, target+1)
    ways = [1] + [0] * target

    for n in ns:
        for i in xrange(n, target+1):
            ways[i] += ways[i-n]
    return ways[target]

def triangular(n):
    return (n * (n + 1)) / 2

def square(n):
    return n ** 2

def pentagonal(n):
    return (n * ((3 * n) - 1)) / 2

def polygonals_below(end, function):
    nums = set()
    for n in count(1):
        num = function(n)
        if num < end:
            nums.add(num)
        else:
            break
    nums = list(nums)
    nums.sort()
    return nums

def triangulars_below(end):
    return polygonals_below(end, triangular)

def squares_below(end):
    return polygonals_below(end, square)

def pentagonals_below(end):
    return polygonals_below(end, pentagonal)

def general_pentagonals_below(end):
    def sub():
        for i in count(0):
            yield (-1) ** i * (i // 2 + 1)
    gen_pens = set() # general pentagonal numbers
    for n in sub():
        gen_pen = pentagonal(n)
        if gen_pen < end:
            gen_pens.add(gen_pen)
        else:
            break
    gen_pens = list(gen_pens)
    gen_pens.sort()
    return gen_pens

def period_cntfrac_sqrt(num):
    """Return the period of the continued fraction of square root.
    >>>cntfrac_sqrt(2)
    >>>1
    >>>cntfrac_sqrt(23)
    >>>4
    >>>cntfrac_sqrt(4)
    >>>0"""
    r = limit = int(sqrt(num))
    if limit * limit == num:
        return 0
    k, period = 1, 0
    while k != 1 or period == 0:
        k = (num - r * r) / k
        r = ((limit + r) / k) * k - r
        period += 1
    return period

def cntfrac_sqrt(num):
    """Return the list of the continued fraction of square root."""
    a = limit = int(sqrt(num))
    if limit * limit == num:
        return []
    b, period = 1, 0
    lst = [limit]
    while b != 1 or period == 0:
        b = (num - a * a) / b
        q = (limit + a) / b
        a = q * b - a
        lst.append(q)
        period += 1
    return lst


if __name__ == '__main__':
    pass

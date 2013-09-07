# ProjectEuler module
# Python 2.7.5


from math import sqrt, floor, log10
from fractions import Fraction, gcd
from itertools import chain, count
import time
from decimal import Decimal

def lcm(numbers): 
    """
    This function is to return the LCM with recursive algorithm. 
    So maybe there are some algorithm faster than this function.
    """
    if len(numbers) == 2: 
        num0 = numbers[0] 
        num1 = numbers[1] 
        return num0 * num1 / gcd(num0, num1) 
    else: 
        for i in range(len(numbers)): 
            return lcm([numbers[0], lcm(numbers[1:])]) 

def num_reverse(num):
    """
    return the reversed number of the integer
    """
    return int(str(num)[::-1])

def prime_factorization(n):
    """
    find which prime numbers multiply together to make the original number
    """
    result = []
    for i in xrange(2, n+1):
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
    """
    calculate the list of the divisors of an integer
    """
    numbers = []
    for i in xrange(1, n+1):
        if n % i == 0:
            numbers.append(i)
    return numbers

def proper_divisors(n):
    """
    calculate the list of the proper divisors of an integer
    """
    numbers = []
    for i in xrange(1, n):
        if n % i == 0:
            numbers.append(i)
            
    return numbers

def count_divisors(n):
    """
    count n's number of divisor numbers
    """
    if n == 1:
        return 0
    m = int(sqrt(n))
    c = 1
    if m * m == n:
        c += 1
        m -= 1
    for i in xrange(2, m+1):
        if n % i == 0:
            c += 2
    return c

def count_proper_divisors(n):
    """
    count n's number of proper divisor numbers
    """
    if n == 1:
        return 0
    m = int(sqrt(n))
    c = 1
    if m * m == n:
        c += 1
        m -= 1
    for i in xrange(2, m+1):
        if n % i == 0:
            c += 2
    return c

"""
def count_proper_divisors2(n):
    # todo: time this function's execute time
    return len(proper_divisors(n))
"""

def sum_divisors(n):
    """
    calculate the sum of divisors
    """
    return sum(proper_divisors(n)) + n

def sum_proper_divisors(n):
    """
    calculate the sum of proper divisors
    """
    return sum(proper_divisors(n))

def is_perfect(n):
    """
    Returns True if integer n is perfect number, otherwise return False.
    """
    if sum_proper_divisors(n) == n:
        return True
    else:
        return False

def is_deficient(n):
    """
    Return True if integer n is deficient number, otherwise return False.
    """
    if sum_proper_divisors(n) < n:
        return True
    else:
        return False

def is_abundant(n):
    """
    Return True if integer n is abundant number, otherwise return False.
    """
    if sum_proper_divisors(n) > n:
        return True
    else:
        return False

def alpha_number(alpha):
    """
    calculate the number of alphabet
    """
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

def num_split(num):
    """
    return the list of the split numbers of integer num
    """
    num = list(str(num))
    return [int(i) for i in num]

def num_digits(num):
    """
    Return the number of digits.
    If num = 0, raise Num Domain Error.
    """
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
    """
    flat the nested list
    """
    return list(chain.from_iterable(nested_list))

def is_pandigital(n, s=9):
    """
    Return True if integer n is pandigital, otherwise return False.
    """
    n = str(n); return len(n) == s and not "1234567890"[:s].strip(n)

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
    """
    Return True if string is palindrome, otherwise return False.
    """
    r_string = string[::-1]
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
    """
    measure the execution time of the function
    """
    start = time.clock()
    func()
    elapsed = time.clock() - start
    print elapsed, "sec"

def totient(x):
    """
    calculate euler's totient function
    """
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

def partition_slow(n):
    """This function is very slow If n is over 100."""
    def sub_partition(n, k):
        if n < 0:
            return 0
        if n <= 1 or k == 1:
            return 1
        s = 0
        for i in xrange(1, k+1):
            s += sub_partition(n-i, i)
        return s
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
    """
    Return the period of the continued fraction of square root.
    >>>cntfrac_sqrt(2)
    >>>1
    >>>cntfrac_sqrt(23)
    >>>4
    >>>cntfrac_sqrt(4)
    >>>0
    """
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
    """
    Return the list of the continued fraction of square root.
    """
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

def is_square(n):
    """
    Return True if integer n is square number, otherwise return False.
    """
    m = int(sqrt(n))
    return m * m == n

def product(L):
    return reduce(lambda x, y: x * y, L)

def is_integer(n):
    """
    Return True if number n is integer, otherwise return False.
    """
    if isinstance(x, float):
        return if x == int(x)
    else:
        raise TypeError, "Input float"

if __name__ == '__main__':
    pass

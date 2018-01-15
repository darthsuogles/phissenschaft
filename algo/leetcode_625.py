""" Minimum factorization
"""

def smallestFactorization(a):
    if a <= 1: return a
    INT_MAX = 1 << 31
    nums = []
    for p in [9,8,7,6,5,4,3,2]:
        while p <= a and 0 == a % p:
            a //= p
            nums.append(p)
        if 1 == a:
            break
    if 1 != a:
        return 0
    val = 0
    for p in nums[::-1]:
        val = val * 10 + p
        if val >= INT_MAX:
            return 0
    return val


print(smallestFactorization(48))
print(smallestFactorization(15))
print(smallestFactorization(1))
print(smallestFactorization(18000000))

''' Form stairs 
'''

def arrangeCoins(n):
    if n <= 1: return n
    from math import sqrt
    delta = sqrt(1 + 8 * n)
    return int(-1 + delta) // 2


def TEST(n):
    s = arrangeCoins(n)
    print(s)
    assert s * (s + 1) <= 2 * n
    assert (s + 1) * (s + 2) > 2 * n


TEST(3)
TEST(4)
TEST(1231)

"""
Monotone increasing digits
"""

def monotoneIncreasingDigits(N):
    if not N: return 0

    postfix_min = N % 10
    pos = 0
    val = N; cnts = 0
    while val:
        cnts += 1; rem = val % 10; val = val // 10
        if postfix_min < rem:
            postfix_min = rem - 1
            pos = cnts
        else:
            postfix_min = rem

    # Preserve the valid monotonically increasing prefix,
    # then replace the rest of the digits with 9s.
    rev_digits = 0
    val = N; cnts = 0
    while val:
        cnts += 1; rem = val % 10; val = val // 10
        curr = None
        if cnts < pos:
            curr = 9
        elif cnts == pos:
            curr = rem - 1
        else:
            curr = rem
        rev_digits = rev_digits * 10 + curr

    # Reverse the digits to obtain the result
    res = 0
    val = rev_digits
    while val:
        rem = val % 10; val = val // 10
        res = res * 10 + rem

    return res


def TEST(N):
    print('orig', N, '=>', monotoneIncreasingDigits(N))


TEST(10)
TEST(332)
TEST(1234)
TEST(1233)
TEST(1232)
TEST(20)
TEST(30)
TEST(1291)

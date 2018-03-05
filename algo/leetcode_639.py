"""
Decode with star
"""

def numDecodings(s):
    if not s: return 0

    MOD = int(1e9 + 7)

    n = len(s)
    tbl = [None] * (n + 1)
    tbl[n] = 1

    for i in range(n - 1, -1, -1):
        cnts = 0
        num = None
        if '*' != s[i]:
            num = ord(s[i]) - ord('0')
            if 0 == num:
                tbl[i] = 0
                continue
            cnts += tbl[i+1]
        else:
            cnts = (9 * tbl[i+1]) % MOD

        if i + 1 < n:
            if '*' != s[i+1]:
                d = ord(s[i+1]) - ord('0')
                if num is not None:
                    num = num * 10 + d
                    if num <= 26:
                        cnts += tbl[i+2]
                else:  # "*d"
                    if d <= 6:  # 2d
                        cnts += tbl[i+2]
                    if d <= 9:  # 1d
                        cnts += tbl[i+2]
            elif 1 == num:  # "1*"
                cnts += (9 * tbl[i+2]) % MOD
            elif 2 == num:  # "2*"
                cnts += (6 * tbl[i+2]) % MOD
            elif num is None:  # "**"
                cnts += (15 * tbl[i+2]) % MOD

        tbl[i] = cnts % MOD

    return tbl[0]


def TEST(s):
    print(numDecodings(s))


TEST("*")
TEST('1*')
TEST('101')

TEST()

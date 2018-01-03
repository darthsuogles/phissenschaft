"""
Distinct subsequence matchings
"""

def numDistinctRef(s, t):
    if not s: return 1 if not t else 0
    if not t: return 1

    rec_tbl = {}
    def find(i, j):
        try: return rec_tbl[(i, j)]
        except: pass
        if j > i: return 0
        if -1 == i: return int(-1 == j)
        if -1 == j: return 1
        a = s[i]; b = t[j]
        cnts = 0
        if a == b:
            cnts += find(i-1, j-1)
        cnts += find(i-1, j)
        return cnts

    return find(len(s) - 1, len(t) - 1)


def numDistinct(s, t):
    if not s: return 1 if not t else 0
    if not t: return 1

    m, n = len(s), len(t)
    rec_tbl = []
    for _ in range(m + 1):
        rec_tbl.append([0] * (n + 1))

    rec_tbl[0][0] = 1

    for i in range(1, m + 1):
        a = s[i-1]
        rec_tbl[i][0] = 1
        for j in range(1, min(n, i) + 1):
            b = t[j-1]
            cnts = 0
            if a == b:
                cnts += rec_tbl[i-1][j-1]
            cnts += rec_tbl[i-1][j]
            rec_tbl[i][j] = cnts

    return rec_tbl[m][n]


def TEST(s, t):
    print(s, t, numDistinct(s, t))

print('-----TEST-CASES------')
TEST('rabbbit', 'rabbit')
TEST('b', 'a')

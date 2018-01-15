"""
Palindromic substring
"""

def countSubstringsDP(s):
    if not s: return 0
    from collections import defaultdict

    char2inds = defaultdict(list)
    palindrom_rec = set()
    for j, ch in enumerate(s):
        for i in char2inds[ch]:
            if i + 1 == j or \
               i + 1 == j - 1 or \
               (i+1, j-1) in palindrom_rec:
                palindrom_rec.add((i, j))
        char2inds[ch] += [j]

    return len(palindrom_rec) + len(s)


def countSubstrings(s):
    if not s: return 0
    cnts = 0
    N = len(s)
    for center in range(2 * N - 1):
        i = center // 2
        j = i + (center % 2)
        while s[i] == s[j]:
            cnts += 1
            i -= 1
            j += 1
            if i < 0 or j >= N:
                break
    return cnts


def TEST(s):
    print(s, countSubstrings(s))


TEST("abc")
TEST("aaa")

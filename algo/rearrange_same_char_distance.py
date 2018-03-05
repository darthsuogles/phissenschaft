"""
Rearrange characters so that the same characters appears d-distance apart
"""
# https://www.geeksforgeeks.org/rearrange-a-string-so-that-all-same-characters-become-at-least-d-distance-away/
def rearrange(s, d):
    if 0 == d: return s
    n = len(s)
    if n < d: return ''

    from collections import Counter
    char_cnts = Counter(s)
    ordered = []
    for ch, cnts in char_cnts.most_common():
        ordered.extend(cnts * ch)

    res = [None] * len(s)
    shift = 0
    j = 0
    for i, ch in enumerate(ordered):
        idx = j * d + shift
        if idx >= n:
            shift += 1
            if d == shift: break
            j = 0
            idx = shift

        res[idx] = ch
        j += 1


    idx = i
    for i in range(n):
        if res[i] is None:
            res[i] = ordered[idx]
            idx += 1

    return res


def TEST(s, d):
    print(rearrange(s, d))


TEST('aaabbcc', 2)
TEST('aaabbcc', 3)

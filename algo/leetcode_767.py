"""
Reorganizing string
"""

def reorganizeString(S):
    if not S: return S
    from collections import Counter
    char_counts = Counter(S)
    n = len(S)
    A = []
    for ch, cnt in char_counts.most_common():
        if cnt > (n + 1) // 2: return ""
        A.extend(cnt * ch)
    res = [None] * n
    # The most common one might contain (n + 1) // 2 elements.
    # Thus they (the leading ones) must be spaced out properly.
    cut_idx = (n // 2) + (n % 2)
    res[::2], res[1::2] = A[:cut_idx], A[cut_idx:]
    return "".join(res)


def TEST(S):
    print(S, "->", reorganizeString(S))

TEST("aab")
TEST("aaab")

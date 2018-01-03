"""
Maximum length subarray
"""

def findLength(A, B):
    """ Binary search with rolling hashing update
    """
    if not A or not B: return 0

    RADIX = 10
    MOD = 10 ** 9 + 7
    Pinv = pow(RADIX, MOD - 2, MOD)  # Fermat's Little Theorem
    def rolling_hash_iter(arr, k):
        if k > len(arr): return
        h = 0; power = 1
        for i, val in enumerate(arr):
            h = (h + val * power) % MOD
            if i + 1 < k:
                power = (power * RADIX) % MOD
            else:
                yield h, i-k+1
                h = (h - arr[i-k+1]) * Pinv % MOD

    def has_substr_match(k):
        if 0 == k: return True
        from collections import defaultdict
        hash2inds = defaultdict(list)
        for h, i in rolling_hash_iter(A, k):
            hash2inds[h].append(i)
        for h, j in rolling_hash_iter(B, k):
            ref = None
            for i in hash2inds[h]:
                ref = ref or B[j:(j+k)]
                if A[i:(i+k)] == ref:
                    return True
        return False

    lo = 0
    hi = min(len(A), len(B)) + 1
    while lo < hi:
        k = (lo + hi) // 2
        if has_substr_match(k):
            lo = k + 1
        else:
            hi = k
    return lo - 1


def findLengthDP(A, B):
    """ Dynamic programming
    """
    if not A or not B: return 0
    if len(A) < len(B):
        A, B = B, A

    max_sublen = [0] * (len(B) + 1)

    tot_max = 0
    for i, a in enumerate(A, 1):
        prev = max_sublen[0]
        for j, b in enumerate(B, 1):
            if a != b:
                prev = max_sublen[j]
                max_sublen[j] = 0
            else:
                curr = 1 + prev
                prev = max_sublen[j]
                max_sublen[j] = curr
                tot_max = max(tot_max, curr)

    return tot_max

def TEST(A, B):
    print(findLength(A, B))


TEST([1,2,3,2,1], [3,2,1,4,7])

TEST([0,0,0,0,0], [0,0,0,0,0])

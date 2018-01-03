"""
Beautiful arrangements
"""

def constructArray(n, k):
    if 1 == k: return list(range(1, n+1))
    assert k > 1
    arr = []
    i = 1; j = n
    # The arrangement adds (k // 2) - 1 diverse diffs
    # Thus if 2 | k, we want the smaller side to go
    # later in the gigsaw pattern, so that the later
    # incremental segments will only add one more
    if 0 == (k % 2):
        for d in range(k // 2):
            arr += [j, i]
            i += 1; j -= 1
    # On the other hand, we let the larger side to go
    # later, so that the incremental segments adds
    # two more diffs instead of one
    else:
        for d in range(k // 2):
            arr += [i, j]
            i += 1; j -= 1        
            
    while i <= j:
        arr.append(i); i += 1
    return arr

def constructArrayBF(N, K):
    assert K > 0
    from collections import defaultdict

    def checkr_gen(arr, tgt):
        n = len(arr) + 1
        k_set = defaultdict(int)
        def decr(key):
            cnt = k_set[key]
            if 0 == cnt or 1 == cnt:
                del k_set[key]
            else:
                k_set[key] = cnt - 1

        for a, b in zip(arr[:-1], arr[1:]):
            k_set[abs(a - b)] += 1
        for i, (a, b) in enumerate(zip(arr[:-1], arr[1:]), 1):
            kna = n - a
            knb = n - b
            kab = abs(a - b)
            k_set[kna] += 1
            k_set[knb] += 1
            decr(kab)
            if len(k_set) == tgt:
                yield arr[:i] + [n] + arr[i:]
            decr(kna)
            decr(knb)

    def kern(n, k):
        if k == 1:
            yield list(range(1, n+1))
            return
        if n < k or k < 1:
            return
        
        assert k > 1
        for p, q in [(n-1, k), (n-1, k-1), (n-1, k-2)]:
            for arr0 in kern(p, q):
                for arr in checkr_gen(arr0, k):
                    yield arr

    for arr in kern(N, K):
        return arr

def TEST(n, k):
    arr = constructArray(n, k)
    from collections import defaultdict
    k_set = defaultdict(int)
    for a, b in zip(arr[:-1], arr[1:]):
        k_set[abs(a - b)] += 1
    print(n, k, arr, k_set)
    assert(len(k_set) == k)

TEST(3, 2)
TEST(3, 1)
TEST(4, 2)
TEST(10, 4)
TEST(5, 4)
TEST(92, 80)

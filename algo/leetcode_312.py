''' Burst ballons
'''

def maxCoins(nums):
    if not nums: return 0
    from collections import defaultdict
    tbl = defaultdict(lambda: 0)

    arr = [1] + nums + [1]
    n = len(arr)
    # Iterate over intervals with length k
    for k in range(2, n):
        # Consider an interval sans boundaries
        #   l, a1, a2, ..., r
        #   ^               ^
        #   |<----- k ----->|
        # As a result, removing any elements within the boundary
        # would not affect anything else outside. Thus the last
        # elements to remove is when there is only (l, a, r).
        for i in range(0, n - k):
            j = i + k
            v_max = 0
            # On the last removal inside the interval
            # Here t is assumed to be the last element left
            # in the interval to pop.
            for t in range(i + 1, j):
                curr = arr[i] * arr[t] * arr[j]
                v_curr = curr + tbl[(i, t)] + tbl[(t, j)]
                v_max = max(v_max, v_curr)

            tbl[(i, j)] = v_max

    return tbl[(0, n - 1)]


def TEST(nums, tgt):
    res = maxCoins(nums)
    if res != tgt:
        print('Error', res, 'but expect', tgt)
    else:
        print('Ok')


TEST([3,1,5,8], 167)
TEST([35,16,83,87,84,59,48,41], 1583373)

''' Sum in range
'''

def sumInRangeBruteForce(nums, queries):
    ''' Compute a weight, this is slow for large queries
    '''
    if not nums: return 0
    n = len(nums)
    idx_weight = [0] * n
    for q in queries:
        i, j = q
        while i <= j:
            idx_weight[i] += 1
            i += 1
    
    tot_sum = 0
    mod = int(1e9 + 7)
    for i, a in enumerate(nums):
        wt = idx_weight[i]
        if wt > 0:
            tot_sum += (a * wt) % mod

    return tot_sum % mod


def sumInRange(nums, queries):
    if not nums: return 0
    n = len(nums)
    prefix_sum = [0] * (n + 1)
    arr_sum = 0
    for i, a in enumerate(nums):
        # Exclusive prefix sum
        prefix_sum[i] = arr_sum
        arr_sum += a
    prefix_sum[n] = arr_sum

    mod = int(1e9 + 7)
    tot_sum = 0
    for q in queries:
        i, j = q
        curr_sum = prefix_sum[j+1] - prefix_sum[i]
        tot_sum += curr_sum % mod

    return tot_sum % mod

def TEST(nums, queries):
    print(sumInRange(nums, queries))


TEST([3, 0, -2, 6, -3, 2], [[0, 2], [2, 5], [0, 5]])

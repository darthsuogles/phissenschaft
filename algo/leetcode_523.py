''' Continuous subarray with sum to multiple of k
'''

def checkSubarraySum(nums, k):
    ''' any integral multiple of k is fine
        k can be any integer
    '''
    if not nums: return False
    n = len(nums)
    if n < 2: return False
    if k < 0: k = -k
    if 1 == k: return True
            
    if 0 != k:
        prefix_modulo = set([nums[0] % k])

    # Updating for all the keys previously seen
    # takes a O(min(n, k)) time
    for i, a in enumerate(nums[1:], 1):
        # Must always check the possbility of zero
        if 0 == a and 0 == nums[i-1]:
            return True
        if 0 == k:
            continue
        mod = a % k
        if k - mod in prefix_modulo:
            return True
        prefix_modulo = set([(x + a) % k for x in prefix_modulo] + [mod])

    return False


def checkSubarraySumScan(nums, k):
    ''' Using prefix sum
    '''
    if not nums: return False
    n = len(nums)
    if n < 2: return False
    if k < 0: k = -k
    if 1 == k: return True

    # Pigeon hole principle: 
    #   at least three of the same prefix sum
    #   with the same modulo, thus at least 
    #   one pair will have a distance greater than one
    if n > 2 * k and k > 0:
        return True

    # Special care for when the modulo op cannot apply
    if 0 == k:
        a_prev = nums[0]
        for a in nums[1:]:
            if 0 == a and 0 == a_prev:
                return True
            a_prev = a

        return False
 
    # Compute prefix sum (incl.) in modulo k
    psum = [0] * (n + 1)
    partial = 0
    for i, a in enumerate(nums, 1):
        partial = (partial + a) % k
        psum[i] = partial        
        if i > 1 and 0 == partial:
            return True

    # Register the first index with a given 
    # prefix sum in modulo k
    psum_first_idx = {}
    for i in range(1, n + 1):
        mod = psum[i]
        try:
            if psum_first_idx[mod] + 1 < i:
                return True
        except: 
            psum_first_idx[mod] = i

    return False
        

def TEST(nums, k, tgt):
    res = checkSubarraySumScan(nums, k)
    if res != tgt:
        print('Error', res, 'but expect', tgt)
    else:
        print('Ok')


TEST([23, 2, 4, 6, 7], 6, True)
TEST([23, 2, 6, 4, 7], 6, True)
TEST([23, 5, 9], 6, False)
TEST([0], 0, False)
TEST([0, 0], 0, True)
TEST([0, 0], -1, True)
TEST([0, 1, 0], -1, True)
TEST([1, 2, 3], 5, True)
TEST([1, 1], 2, True)

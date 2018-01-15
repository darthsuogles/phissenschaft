''' Sliding window median
'''

def medianSlidingWindow(nums, k):
    ''' Keep track of a sorted array
    '''
    if len(nums) < k: return []
    from bisect import bisect_left

    if 0 == k % 2:
        def get_median(window, k):
            return (window[k//2] + window[k//2 - 1]) / 2.0
    else:
        def get_median(window, k):
            return float(window[k//2])
        
    window = sorted(nums[:k])
    medians = [get_median(window, k)]    
    for i, a in enumerate(nums[k:], k):
        # We are guaranteed to have this position
        j_out = bisect_left(window, nums[i-k])  
        window.pop(j_out)
        j_in = bisect_left(window, a)
        window.insert(j_in, a)
        medians.append(get_median(window, k))

    return medians


def TEST(nums, k):
    print(medianSlidingWindow(nums, k))

TEST([1,4,2,3], 4)
TEST([1,3,-1,-3,5,3,6,7], 2)


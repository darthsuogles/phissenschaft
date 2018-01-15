""" Find 132 pattern in sequence
"""

def find132pattern(nums):
    n = len(nums)
    if n < 3: return False
    inc_back = []
    cand = None
    for a in nums[::-1]:
        if cand is not None and cand > a:
            return True
        while inc_back:
            b = inc_back[-1]
            if b > a: break
            if b != a: cand = b
            inc_back.pop(-1)            
        inc_back.append(a)
        
    return False
        

def find132patternPartialSort(nums):
    if len(nums) < 3: return False
    n = len(nums)
    from bisect import bisect_left
    inc_right = []
    right_maxmin = [None] * n
    for i in range(n-1, -1, -1):
        a = nums[i]
        k = bisect_left(inc_right, a)
        if k > 0:
            right_maxmin[i] = inc_right[k-1]
        inc_right.insert(k, a)

    left_min = nums[0]
    for i, a in enumerate(nums[1:], 1):
        if left_min > a: 
            left_min = a; continue
        if left_min == a: continue
        b = right_maxmin[i]
        if b is None: continue
        if b > left_min:
            return True

    return False

def TEST(nums, tgt):
    res = find132pattern(nums)
    if res != tgt:
        print('Error', res, 'but expect', tgt)
    else:
        print('Ok')


TEST([1,2,3,4], False)
TEST([3,1,4,2], True)
TEST([-1,3,2,0], True)
TEST([3,5,0,3,4], True)
TEST([-2, 1, 1], False)

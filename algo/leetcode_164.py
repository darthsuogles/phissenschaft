''' Maximum distance of consecutive elements in sorted order
    Implmentation should not rely on sorting
'''

def maximumGapRef(nums):
    arr = sorted(nums)
    return max(map(lambda e: abs(e[0] - e[1]), zip(arr[1:], arr[:-1])))


def maximumGapRec(nums):
    ''' Recursively find min max and gap
    '''
    if not nums: return 0
    if len(nums) < 2: return 0

    def getMinMaxGap(nums):    
        n = len(nums)
        if 1 == n: return (nums[0], nums[0], 0)

        pivot = nums[0]
        arr_lo = [a for a in nums[1:] if a <= pivot]
        arr_hi = [a for a in nums if a > pivot]

        if arr_lo:
            min_lo, max_lo, gap_lo = getMinMaxGap(arr_lo)
        else:
            min_lo, max_lo, gap_lo = pivot, pivot, 0

        if arr_hi:
            min_hi, max_hi, gap_hi = getMinMaxGap(arr_hi)
        else:
            min_hi, max_hi, gap_hi = pivot, pivot, 0

        gap_curr = max(max(gap_hi, gap_lo), 
                       max(pivot - max_lo, min_hi - pivot))

        return (min_lo, max_hi, gap_curr)

    return getMinMaxGap(nums)[-1]


def maximumGap(nums):
    if not nums: return 0
    n = len(nums)
    if n < 2: return 0
    
    v_max = nums[0]; v_min = nums[0]
    for a in nums:
        v_max = max(v_max, a)
        v_min = min(v_min, a)

    if v_max == v_min: return 0
    from math import ceil    
    gap_size = int(ceil((v_max - v_min) * 1.0 / n))  # backward compat

    slots = [None] * (n+1)
    s_min = v_min
    for i in range(n+1):        
        if s_min > v_max: break
        slots[i] = (s_min + gap_size, s_min, 0)
        s_min += gap_size

    slots = slots[:i]

    for a in nums:
        i = (a - v_min) // gap_size
        s_min, s_max, s_cnt = slots[i]
        slots[i] = (min(s_min, a), max(s_max, a), s_cnt + 1)

    nzs = [(s_min, s_max) for s_min, s_max, s_cnt in slots if s_cnt > 0]
    assert len(nzs) > 1
    return max(a1 - b0 for (a1, b1), (a0, b0) in zip(nzs[1:], nzs[:-1]))
        

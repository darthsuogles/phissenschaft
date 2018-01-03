''' K-th largest
'''

def findKthLargest(nums, k):
    
    if not nums: return None
    from random import randint

    init = 0; fini = len(nums) - 1
    while True:    
        if init > fini: return None        
        if init <= fini and 1 == k:
            return nums[init]

        # The quick-sort manipulation
        t = randint(init, fini)
        nums[init], nums[t] = nums[t], nums[init]
        pivot = nums[init]
        i = init; j = fini
        while True:
            while i < fini and nums[i] >= pivot: i += 1
            while nums[j] < pivot: j -= 1
            # Always make sure j points to a number >= pivot
            if j <= i: break
            nums[j], nums[i] = nums[i], nums[j]
            i += 1; j -= 1
            
        m = j - init + 1
        if m == k:
            return pivot
        if m < k:
            init = j + 1
            k -= m
        else:
            init = init + 1
            fini = j            
        


def TEST(nums, k, tgt):
    res = findKthLargest(nums, k)
    if res != tgt:
        print('Error', res, 'but expect', tgt)
    else:
        print('Ok')
    

TEST([3,2,1,5,6,4], 2, 5)
TEST([2, 1], 2, 1)
TEST([3,1,2,4], 2, 3)

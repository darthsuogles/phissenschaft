''' Next greater number
''' 

def nextGreaterElement(findNums, nums):
    if not findNums: return []
    n = len(nums)
    if 0 == n: return []

    from collections import defaultdict
    tbl_gt_next = defaultdict(lambda: -1)
    buf = [nums[0]]
    for a in nums[1:]:
        buf_next = []
        for b in buf:
            if b < a:
                tbl_gt_next[b] = a
            else:
                buf_next += [b]        

        del buf; buf = buf_next; buf += [a]

    return [tbl_gt_next[a] for a in findNums]
    

def TEST(nums1, nums2):
    res = nextGreaterElement(nums1, nums2)
    print(res)


TEST([4,1,2], [1,3,4,2])
TEST([2,4], [1,2,3,4])
TEST([1,3,5,2,4], [6,5,4,3,2,1,7])

"""
Longest consecutive sequence

https://leetcode.com/problems/longest-consecutive-sequence/description/
"""

def longestConsecutive(nums):
    """ Merge intervals, as a connected component problem """
    if not nums: return 0
    tbl_fini = {}
    tbl_init = {}
    max_len = 0
    for a in nums:
        if a in tbl_fini or a in tbl_init:
            continue
        if a + 1 in tbl_init:
            j = tbl_init[a + 1]
            del tbl_init[a + 1]
        else:
            j = a
        if a - 1 in tbl_fini:
            i = tbl_fini[a - 1]
            del tbl_fini[a - 1]
        else:
            i = a
        tbl_init[i] = j
        tbl_fini[j] = i
        max_len = max(max_len, j - i + 1)

    return max_len

def longestConsecutiveRef(nums):
    if not nums: return 0
    nums_ord = sorted(nums)
    max_len = 0
    prev = nums_ord[0]
    curr_len = 1
    for a in nums_ord[1:]:
        if prev == a:
            continue
        elif prev + 1 == a:
            curr_len += 1
        else:
            max_len = max(max_len, curr_len)
            curr_len = 1
        prev = a

    return max_len

def TEST(nums):
    res = longestConsecutive(nums)
    tgt = longestConsecutiveRef(nums)
    print(res, res == tgt)

print('----TEST CASES----')
TEST([100, 4, 200, 1, 3, 2])
TEST([-7,-1,3,-9,-4,7,-3,2,4,9,4,-9,8,-7,5,-1,-7])

"""
Non-decreasing array with one modification
"""

def checkPossility(nums):
    if not nums: return True
    INT_MIN = int(-1e11)
    INT_MAX = int(+1e11)
    nums = [INT_MIN] + nums + [INT_MAX]
    n = len(nums); i = 1
    while i + 1 < n:
        p, v, q = nums[i-1], nums[i], nums[i+1]
        if p <= v and v <= q:
            i += 1; continue
        # To determine the culprit we need context
        if p > q: # e.g. (p,v,q) is (3,4,1)
            nums[i+1] = v
        else: # e.g. (p,v,q) is (3,2,4)
            nums[i] = p
        break

    while i + 1 < n:
        if nums[i] > nums[i+1]:
            return False
        i += 1

    return True


def TEST(nums, ref):
    tgt = checkPossility(nums)
    assert tgt == ref, (nums, 'ref', ref, '!=', tgt)

TEST([4,2,3], True)
TEST([4,2,1], False)
TEST([3,4,2,3], False)
TEST([2,3,3,2,4], True)

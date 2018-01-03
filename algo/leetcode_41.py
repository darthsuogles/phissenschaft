''' First missing positive
'''

def firstMissingPositiveSort(nums):
    if not nums: return 1
    nums_pos_uniq = sorted(set([a for a in nums if a > 0]))
    for i, a in enumerate(nums_pos_uniq, 1):
        if i < a: return i
    return 1 + len(nums_pos_uniq)


def firstMissingPositive(nums):
    if not nums: 
        return 1
    n = len(nums)  # first missing one is <= n + 1
    # The missing number must be <= n
    for i, a in enumerate(nums):
        if a < 1:  # ignore non-positive numbers
            continue
        if a == i + 1:  # the number that matches the index
            continue
        j = a - 1
        nums[i] = -1  # don't know whether it will be filled
        while 0 <= j and j < n:
            if nums[j] == j + 1: break
            k = nums[j] - 1
            nums[j] = j + 1
            j = k
                
    for i, a in enumerate(nums, 1):
        if i != a: return i
    return n + 1


def TEST(nums):
    print(firstMissingPositiveSort(nums))

TEST([1,2,0])
TEST([3,4,-1,1])

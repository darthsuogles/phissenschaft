"""
Subsets
"""

def subsets(nums):
    power_set = [[]]
    if not nums: return power_set

    for a in nums:
        max_idx = len(power_set)
        for elem in power_set[:max_idx]:
            power_set.append(elem + [a])

    return power_set


def TEST(nums):
    for elem in subsets(nums):
        print(elem)


TEST([1,2,3])

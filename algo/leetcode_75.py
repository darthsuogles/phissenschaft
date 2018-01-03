
def sortColors(nums, K):
    i = 0
    for k in range(0, K):
        j = len(nums) - 1
        while i < j:
            while i < j and nums[i] == k: i += 1
            while i < j and nums[j] != k: j -= 1
            if i == j: break
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
            j -= 1

        # Clean up the last ones
        while i < len(nums) and nums[i] == k: i += 1
        if i == len(nums): break

    return nums


def TEST(nums):
    K = len(set(nums))
    sortColors(nums, K)
    for a, b in zip(nums[:-1], nums[1:]):
        if a > b:
            print("ERROR"); return
    print("OK")

TEST([1,1,0,2,0,2])
TEST([2,0,1])

import numpy as np
for K in range(2, 10):
    TEST(np.random.randint(0, K, 100))

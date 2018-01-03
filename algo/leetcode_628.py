""" Maximum product of three
"""

def maximumProduct(nums):
    if len(nums) < 3: return 0
    nums = sorted(nums)
    a, b = nums[:2]
    c, d, e = nums[-3:]
    post_max = c * d * e
    if b < 0:
        return max(a * b * e, post_max)
    else:
        return post_max


def TEST(nums):
    print(maximumProduct(nums))

TEST([-3,-2,-1,1,2,3])

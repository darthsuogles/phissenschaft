"""
Single number III
"""

def singleNumber(nums):
    if not nums: return []
    val = 0
    for a in nums:
        val = val ^ a

    # Separate the numbers into two groups
    # based on the bit mask
    bit_mask = 1
    while True:
        if bit_mask & val != 0:
            break
        bit_mask <<= 1

    vbs = [0, 0]
    for a in nums:
        vbs[(bit_mask & a != 0)] ^= a

    return vbs

def TEST(nums):
    print(singleNumber(nums))

TEST([1, 2, 1, 3, 2, 5])

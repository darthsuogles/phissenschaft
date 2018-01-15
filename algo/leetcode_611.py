"""
Valid triangle
"""


def triangleNumberRef(nums):
    nums = sorted(nums)
    cnts = 0
    for i, a in enumerate(nums):
        for j, b in enumerate(nums[(i + 1):], i + 1):
            for c in nums[(j + 1):]:
                if c >= a + b:
                    break
                cnts += 1

    return cnts


def triangleNumberBS(nums):
    nums = sorted(nums)
    cnts = 0
    N = len(nums)
    p, q = 0, N - 1
    for p in range(0, N - 1):
        for q in range(N - 1, p + 1, -1):
            _first = nums[p]
            tgt = nums[q] - _first

            # Short-circuit
            if tgt < _first:
                cnts += q - p - 1
                continue

            # Do a binary search (bisect_right)
            i, j = p + 1, q
            while i < j:
                k = (i + j) // 2
                if nums[k] <= tgt:
                    i = k + 1
                else:
                    j = k

            # Now, index i points to the position of the first desired value
            # From there till the one before position q, the values will help
            # us form the triangle.
            cnts += q - i

    return cnts


def triangleNumber(nums):
    N = len(nums)
    if N < 3: return 0
    nums = sorted(nums)
    cnts = 0
    for i in range(N - 2):
        a = nums[i]
        if a < 1: continue
        k = i + 2
        for j in range(i + 1, N - 1):
            up_bnd = a + nums[j]
            while k < N:
                if nums[k] >= up_bnd:
                    break
                k += 1
            m = k - j - 1
            cnts += m
            if k == N:
                cnts += m * (m - 1) // 2
                break


    return cnts

def TEST(nums):
    ref = triangleNumberRef(nums)
    tgt = triangleNumber(nums)
    print('ref', ref, 'tgt', tgt)

print('-------TEST-CASES--------')
TEST([2,2,3,4])
TEST([1,1,3,4])
TEST([0,1,0,1])
TEST([1,2,3,4,5,6])

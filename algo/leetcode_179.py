"""
Largest number
"""

def largestNumber(nums):
    if not nums: return ""
    import functools
    def concat(a, b): return int(str(a) + str(b))
    def cmp_fn(a, b): return concat(a, b) - concat(b, a)
    num_ord = sorted(nums, key=functools.cmp_to_key(cmp_fn), reverse=True)
    for idx, num in enumerate(num_ord): if num != 0: break
    return ''.join(map(str, num_ord[idx:]))



def TEST(nums):
    print(largestNumber(nums))


TEST([3, 30, 34, 5, 9])
TEST([0, 0])

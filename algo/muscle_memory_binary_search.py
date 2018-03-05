"""
Build muscle memory: binary search
"""

# https://www.topcoder.com/community/data-science/data-science-tutorials/binary-search/

def binary_search(nums, predicate):
    """
    The returned index points to the position in `nums` such that
    if we insert the new element to that position, we will
    satisfy the `predicate` evaluation assumption.

    That is, assuming the input is sorted based on the predicate
    evaluation results: F...T
    """
    if not nums:
        return 0

    # If the last element evaluates to F, we immedately return
    n = len(nums)
    if not predicate(nums[-1]):
        return n

    # In the end, `i` will point to the first element where
    # `predicate(nums[i])` => T
    i = 0; j = n - 1;
    # Must NOT use `i <= j`, as the update will go into infinite loop
    while i < j:
        k = i + (j - i) // 2
        if predicate(nums[k]):
            j = k
        else:
            i = k + 1
        # # Or it could be written this way:
        # if not predicate(nums[k]):
        #     i = k + 1
        # else:
        #     j = k

    return i

def bisect_left(nums, val):
    def predicate(a):
        return val <= a
    return binary_search(nums, predicate)

def bisect_right(nums, val):
    def predicate(a):
        return val < a
    return binary_search(nums, predicate)

def TEST(nums, val):
    import bisect
    i_ref = bisect.bisect_left(nums, val)
    i_tgt = bisect_left(nums, val)
    assert i_ref == i_tgt, ('ref', i_ref, 'tgt', i_tgt)
    i_ref = bisect.bisect_right(nums, val)
    i_tgt = bisect_right(nums, val)
    assert i_ref == i_tgt, ('ref', i_ref, 'tgt', i_tgt)
    print('PASS')

def test_common():
    nums = [1,1,2,2,4,4,4,7]
    TEST(nums, 0)
    TEST(nums, 1)
    TEST(nums, 7)
    TEST(nums, 3)
    TEST(nums, 9)

test_common()

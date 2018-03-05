''' Search in rotated sorted array
'''

def search(nums, target):
    n = len(nums)
    if 0 == n:
        return -1

    # Relies on modular arithmetic
    i = 0; j = n - 1;
    while i != j:
        k = (i + ((j - i) % n) // 2) % n
        if nums[k] > nums[j]:
            i, k, j = j, i, k
        elif nums[k] < nums[i]:
            i, k, j = k, j, i

        if target == nums[i]: return i
        if target == nums[k]: return k
        if target == nums[j]: return j

        # Every time the sequence is partitioned into four segments
        if nums[k] < target and target < nums[j]:
            i, j = (k + 1) % n, j
        elif nums[j] < target or target < nums[i]:
            i, j = (j + 1) % n, i
        elif nums[i] < target and target < nums[k]:
            i, j = (i + 1) % n, k

    if target == nums[i]:
        return i
    else:
        return -1


def search_complex(nums, target):
    if not nums: return -1
    # First find the smallest pivot
    def find_pivot():
        i = 0; j = len(nums) - 1
        if nums[i] < nums[j]:
            return 0  # sorted array
        pivot = None
        while i + 1 < j:
            k = (i + j) // 2
            vi = nums[i]; vj = nums[j]
            a = nums[k]
            if a >= vi:
                i = k; continue
            if a <= vj:
                if k > 0 and nums[k-1] > a:
                    return k
                j = k; continue

        if pivot is None:
            if nums[i] < nums[j]:
                return i
            else:
                return j

    pivot = find_pivot()
    print('pivot', pivot, nums[pivot])

    def bin_search(arr):
        if not arr: return -1
        i = 0; j = len(arr) - 1
        while i + 1 < j:
            k = (i + j) // 2
            a = arr[k]
            if a == target:
                return k
            if a < target:
                i = k
            else:
                j = k

        if arr[i] == target:
            return i
        if arr[j] == target:
            return j
        return -1

    i = bin_search(nums[:pivot])
    if -1 == i:
        j = bin_search(nums[pivot:])
        if -1 == j:
            return -1
        return pivot + j
    return i


def TEST(arr, tgt, not_found=False):
    i = search(arr, tgt)
    print('TEST CASE', arr, tgt, 'res', i)
    assert (-1 == i and not_found) or (-1 != i and arr[i] == tgt), i



TEST([4,5,5,6,7,0,1,2], 2)
TEST([4,5,6,7,1,2,4], 5)
TEST([1], 0, not_found=True)
TEST([3,1], 1)
TEST([1], 1)
TEST([1,3], 3)
TEST([4,5,6,7,0,1,2], 0)
TEST([4,5,6,7,0,1,2], 1)

''' Array intersection
'''

def intersect(nums1, nums2):
    if nums1 is None or [] == nums1:
        return []
    if nums2 is None or [] == nums2:
        return []

    i = 0; j = 0
    m = len(nums1); n = len(nums2)
    nums1 = sorted(nums1)
    nums2 = sorted(nums2)

    res = []
    while i < m and j < n:
        a = nums1[i]
        b = nums2[j]
        if a == b:
            res += [a]
            i += 1; j += 1
        elif a < b:
            i += 1
        else:
            j += 1

    return res


def TEST(nums1, nums2, tgt):
    res = intersect(nums1, nums2)
    if sorted(tgt) != sorted(res):
        print("Error", res, 'but expect', tgt)
    else:
        print("Ok")


TEST([1, 2, 2, 1], [2, 2], [2, 2])

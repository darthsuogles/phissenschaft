''' Median of two sorted arrays
''' 

def findMedianSortedArrays(nums1, nums2):
    ''' Invoke a find kth-smallest method 
    '''
    if not nums1 and not nums2: return None

    def find_kth_smallest(nums1, nums2, k):
        ''' 1 <= k <= m + n
        '''
        m = len(nums1); n = len(nums2)
        assert m + n >= k
        if m > n: 
            return find_kth_smallest(nums2, nums1, k)

        # Base cases
        if 0 == m:
            return nums2[k-1]
        if 1 == k:
            return min(nums1[0], nums2[0])

        # Find lower half of the shorter array
        sa = min(m, k // 2); sb = k - sa
        a = nums1[sa-1]
        b = nums2[sb-1]
        if a == b: # found the k-th smallest
            return a
        if a < b:
            return find_kth_smallest(nums1[sa:], nums2, k-sa)
        else:
            return find_kth_smallest(nums1, nums2[sb:], k-sb)

    m = len(nums1); n = len(nums2); k = (m + n) // 2
    # For odd (m + n), take the (k + 1)-th which is the center
    if k + k != m + n:
        med = find_kth_smallest(nums1, nums2, k+1)
    # For even (m + n), must take k-th and (k + 1)-th and average
    else:
        med1 = find_kth_smallest(nums1, nums2, k)
        med2 = find_kth_smallest(nums1, nums2, k+1)
        med = (med1 + med2) / 2.0

    return float(med)


def TEST(nums1, nums2):
    def get_median(nums):
        if not nums: return None
        n = len(nums)
        if 0 == n % 2:
            return (nums[n // 2] + nums[n // 2 - 1]) / 2.0
        else:
            return float(nums[n // 2])

    tgt = get_median(sorted(nums1 + nums2))
    res = findMedianSortedArrays(nums1, nums2)

    if tgt != res:
        print('Error', res, 'but expect', tgt)
    else:
        print('Ok')


TEST([1,2], [3,4])
TEST([1,3], [2])
TEST([1,3,7], [2,5,8])
TEST([1,2], [3,4,5,6])
TEST([1,2], [1,2,3])
TEST([1,3], [2,4,5,6])

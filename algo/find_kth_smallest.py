
def find_kth(nums, k):
    """ Find the k-th (1-based) smallest element in linear time
    """
    n = len(nums)
    assert n >= k and k > 0
    # Base case
    if 1 == n: return nums[0]

    # Find the best half to continue the search
    # based on the chosen element "p"
    p = nums[-1]

    # When the loop terminates, i will point to the
    # last element that is greater than or equal to "p"
    i = 0; j = n - 2
    while i <= j:  # with equal, we resolve boundary condition
        while j >= 0 and nums[j] >= p:
            j -= 1
        while nums[i] < p:
            i += 1
        if i < j:
            nums[i], nums[j] = nums[j], nums[i]

    # We know that "p" is the (i+1)-th largest element
    if i + 1 == k:
        return nums[-1]  # target is found
    if i >= k:
        return find_kth(nums[:i], k)
    else:
        return find_kth(nums[i:-1], k - i - 1)


def TEST(nums, idx):
    assert find_kth(nums, idx) == sorted(nums)[idx - 1]

TEST([1,2,3,4,5,6], 3)
TEST([1,1,1,3,4,2], 3)

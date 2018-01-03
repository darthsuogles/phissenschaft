""" Find peak element
"""

def findPeakElement(nums):
    if not nums: return -1
    n = len(nums)
    i, j = 0, n-1
    while i + 1 < j:
        k = (i + j) // 2
        gtl = True if 0 == k else nums[k-1] < nums[k]
        gtr = True if k + 1 == n else nums[k+1] < nums[k]
        if gtl and gtr: return k
        if gtl: i = k
        else: j = k
    return j if nums[i] < nums[j] else i

def findPeakElementRef(nums):
    if not nums: return -1
    n = len(nums)
    if 1 == n: return 0
    if nums[0] > nums[1]: return 0
    if nums[-1] > nums[-2]: return n-1
    for i in range(1, n-1):
        if nums[i-1] < nums[i] and nums[i] > nums[i+1]:
            return i
    return n-1

print(findPeakElement([1,2,3,1]))
print(findPeakElement([1,2]))

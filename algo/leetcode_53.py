''' Maximum subarray
''' 

def maxSubArray(nums):
    if not nums: return 0
    curr_max = nums[0]
    curr = nums[0]
    for a in nums[1:]:
        if curr < 0:
            curr = 0
        curr += a
        curr_max = max(curr, curr_max)

    return curr_max


maxSubArray([-2,1,-3,4,-1,2,1,-5,4])
maxSubArray([-1,-1,-1,-1])
maxSubArray([-1,-1,0,-1])

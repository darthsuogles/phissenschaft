''' Move all zeros to the end of the array
'''

def moveZeros(nums):
    if not nums: return
    i = 0; j = 0
    n = len(nums)
    for j, val in enumerate(nums):
        if 0 == val: 
            continue
        nums[i] = nums[j]
        i += 1
    while i < n:
        nums[i] = 0; i += 1


nums = [0, 1, 0, 3, 12]
moveZeros(nums)
            
        

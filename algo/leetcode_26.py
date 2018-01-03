''' Remove duplicates
''' 

def removeDuplicates(nums):
    if not nums: return 0
    n = len(nums)
    if n < 2: return n
    i = 1; j = 1
    while j < n:
        if nums[j] != nums[j-1]: 
            nums[i] = nums[j]
            i += 1
        j += 1
        
    return i

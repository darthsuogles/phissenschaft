""" Connected components in permutation
"""

def arrayNesting(nums):
    if not nums: return 0
    visited = set()
    curr_size = 0
    max_size = 0
    for i, j in enumerate(nums):        
        if i in visited: continue
        while True:
            if i in visited:
                max_size = max(max_size, curr_size)
                curr_size = 0
                break
            visited.add(i)
            curr_size += 1
            i, j = j, nums[j]
            
    return max_size


def TEST(nums):
    print(arrayNesting(nums))

TEST([5,4,0,3,1,6,2])
TEST([0, 1, 2])

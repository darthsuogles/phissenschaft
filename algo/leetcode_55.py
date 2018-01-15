''' Jump game
'''

def canJump(nums):
    if not nums: return True
    n = len(nums)
    if n <= 1: return True

    tbl_can_jump = [False] * n
    tbl_can_jump[n-1] = True

    for i in range(n-2, -1, -1):            
        k = nums[i]
        if k == 0:
            tbl_can_jump[i] = False
            continue
        curr_can_jump = False
        for j in range(min(i+k, n-1), i, -1):
            if tbl_can_jump[j]:
                curr_can_jump = True
                break
        tbl_can_jump[i] = curr_can_jump
    
    return tbl_can_jump[0]


def canJump(nums):
    n = len(nums)
    if n <= 1: return True
    
    i_max = 0
    for i in range(n):
        i_max = max(i_max, i + nums[i])        
        if i_max == i:
            return False
        if i_max + 1 >= n:
            return True        

    return False


def TEST(nums):
    print(canJump(nums))

TEST([2,3,1,1,4])
TEST([3,2,1,0,4])

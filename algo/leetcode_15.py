''' Find all triplets that sum to zero
'''

def threeSumBruteForce(nums):
    if not nums: return []
    n = len(nums)
    res = set()
    for i in range(n-2):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                a, b, c = nums[i], nums[j], nums[k]
                if 0 == a + b + c:
                    tp = sorted((a, b, c))
                    res.add(tuple(tp))
    
    return list(res)


def threeSum(nums):
    ''' Using pointer
    '''
    if not nums: return []
    n = len(nums)
    nums.sort()

    idx = 0
    triplets = []
    target = 0

    while idx + 2 < n:
        if 0 < idx and nums[idx-1] == nums[idx]: 
            idx += 1; continue
        v0 = nums[idx]
        curr_target = target - v0
        
        i = idx + 1; j = n - 1
        while i < j:
            a = nums[i]; b = nums[j]        

            # If approximation is needed, use this to judge
            if a + b == curr_target:
                triplets.append((v0, a, b))
                while i < j:
                    if nums[i] != a: break
                    i += 1
                while i < j:
                    if nums[j] != b: break
                    j -= 1

            elif a + b < curr_target:
                i += 1
            else:
                j -= 1
        
        idx += 1
                
    return triplets

def threeSumBinarySearch(nums):    
    ''' Requires a binary search for each iteration
    '''
    if nums is None:
        return []

    from bisect import bisect_right
    nums = sorted(nums)
    res = []

    n = len(nums)
    i = 0; j = n - 1
    target = 0
    while i + 1 < j:
        a = nums[i]; b = nums[j]        
        # k is the insertion position 
        # elements in i+1 ... k are all <= the target
        k = bisect_right(nums, target - (a + b), i+1, j)        
        if i + 1 == k:  # a+b too big, decrease b
            while i < j:
                if nums[j] != b: break
                j -= 1
            if i >= j: break
            i = 0  # the lower bound is smaller
            continue

        if nums[k-1] + a + b == 0:
            res += [(a, nums[k-1], b)]

        while i < j:
            if nums[i] != a: break
            i += 1   

    return res


def TEST(nums, tgt):
    tgt = sorted(list(set(tuple(sorted(l)) for l in tgt)))
    res = sorted(threeSum(nums))
    if res != tgt:
        print("ERROR")
        print("\tRES", res)
        print("\tTGT", tgt)
    else:
        print("OK")

TEST([-1, 0, 1, 2, -1, -4], [[-1,0,1], [-1,-1,2]])
TEST([-4,-2,-2,-2,0,1,2,2,2,3,3,4,4,6,6], 
     [[-4,-2,6],[-4,0,4],[-4,1,3],[-4,2,2],[-2,-2,4],[-2,0,2]])
TEST([-4,-2,1,-5,-4,-4,4,-2,0,4,0,-2,3,1,-5,0],
     [[-5,1,4],[-4,0,4],[-4,1,3],[-2,-2,4],[-2,1,1],[0,0,0]])

''' Get largest mutually divisible subset
'''

tbl = {}

def largestDivisibleSubsetRec(nums):
    if nums is None: return []
    n = len(nums)
    if 0 == n: return []
    if 1 == n: return nums

    try: return tbl[tuple(nums)]
    except: pass

    res = [nums[0]]
    for i, a in enumerate(nums[1:], 1):
        cands = []
        for b in nums[:i]:
            if (0 == a % b) or (0 == b % a):
                cands += [b]
        curr = largestDivisibleSubsetRec(cands) + [a]
        tbl[tuple(cands + [a])] = curr
        if len(curr) > len(res):
            res = curr
            
    return res


def largestDivisibleSubset(nums):
    tbl = {-1: set()}  # ensure always non-empty
    for i, a in enumerate(sorted(nums)):
        tbl[a] = max((tbl[d] for d in tbl if 0 == a % d), key=len) | {a}
    return list(max(tbl.values(), key=len))


def TEST(nums):
    res = largestDivisibleSubset(nums)
    print(res)


TEST([1,2,3])
TEST([1,2,4,8])
        

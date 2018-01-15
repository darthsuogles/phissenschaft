""" Target sum
"""

def findTargetSumWaysRec(nums, S):
    if not nums: return int(0 == S)
    tbl = {}
    def find_num_ways(nums, S, tbl):
        if not nums: return int(0 == S)
        n = len(nums)
        try: return tbl[(n, S)]
        except: pass
        d = nums[-1]; nums = nums[:-1]
        pos_cnts = find_num_ways(nums, S + d, tbl)
        neg_cnts = find_num_ways(nums, S - d, tbl)
        tbl[(n, S)] = pos_cnts + neg_cnts
        return pos_cnts + neg_cnts

    return find_num_ways(nums, S, tbl)


def findTargetSumWays(nums, S):
    if not nums: return int(0 == S)
    N, W = len(nums), sum(nums)
    from collections import defaultdict

    last_vals = {0: 1}
    for i, a in enumerate(nums):
        curr_vals = defaultdict(int)
        for val, cnts in last_vals.items():
            curr_vals[val + a] += cnts
            curr_vals[val - a] += cnts

        last_vals = curr_vals
    
    try: return curr_vals[S]
    except: return 0
    

#print(findTargetSumWays([1,1,1,1,1], 3))
print(findTargetSumWays([1000], -1000))
print(findTargetSumWays([0,0,0,0,0,0,0,0,1], 1))

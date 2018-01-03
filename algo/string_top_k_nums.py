''' Get largest k numbers in a string   
'''

def top_k_from_str(s, k):
    if not s: return []
    from heapq import heappush, heappop, nlargest

    sign = 1
    top_nums = []

    num = None
    for i, ch in enumerate(s):        
        if '0' <= ch <= '9':
            if num is None: 
                if i > 0 and s[i-1] == '-':
                    sign = -1
                num = 0
            num = num * 10 + int(ch)
        elif num is None:
            # There are only non-numeric chars, ignore
            continue
        else:
            heappush(top_nums, sign * num)
            if len(top_nums) > k:
                heappop(top_nums)
            num = None
            sign = 1
    
    if num is not None:
        heappush(top_nums, sign * num)
        
    print(nlargest(k, top_nums))


top_k_from_str('dfs-fs-980sdf-123poier110poipoikkj100', k=3)

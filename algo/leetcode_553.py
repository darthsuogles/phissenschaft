""" Optimal division
"""

def optimalDivision(nums):
    n = len(nums)
    if 0 == n: return ""
    if 1 == n: return str(nums[0])

    div_min_tbl = {}
    div_max_tbl = {}
    max_expr_tbl = {}
    min_expr_tbl = {}
    for i, a in enumerate(nums): 
        div_min_tbl[(i, i)] = float(a)
        div_max_tbl[(i, i)] = float(a)
        max_expr_tbl[(i, i)] = str(a)
        min_expr_tbl[(i, i)] = str(a)

    for k in range(1, n+1):
        for i in range(0, n-k):
            j = i + k
            curr_min = 10000000; curr_max = -1
            curr_min_expr = None; curr_max_expr = None
            for t in range(i, j):
                # Update max
                curr = div_max_tbl[(i,t)] / div_min_tbl[(t+1,j)]
                if curr > curr_max:
                    curr_max = curr
                    a_expr = max_expr_tbl[(i, t)]
                    b_expr = min_expr_tbl[(t+1, j)]
                    if t + 1 < j:
                        curr_max_expr = "{}/({})".format(a_expr, b_expr)
                    else:
                        curr_max_expr = "{}/{}".format(a_expr, b_expr)

                # Update min
                curr = div_min_tbl[(i,t)] / div_max_tbl[(t+1, j)]
                if curr < curr_min:
                    curr_min = curr
                    a_expr = min_expr_tbl[(i, t)]
                    b_expr = max_expr_tbl[(t+1, j)]
                    if t + 1 < j:
                        curr_min_expr = "{}/({})".format(a_expr, b_expr)
                    else:
                        curr_min_expr = "{}/{}".format(a_expr, b_expr)
                    
                    
            div_min_tbl[(i, j)] = curr_min
            div_max_tbl[(i, j)] = curr_max
            max_expr_tbl[(i, j)] = curr_max_expr
            min_expr_tbl[(i, j)] = curr_min_expr

    return max_expr_tbl[(0, n-1)]


print(optimalDivision([1000, 100, 10, 2]))
print(optimalDivision([6,2,3,4,5]))

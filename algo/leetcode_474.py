"""
Find maximum number of strings can be formed 
"""

def findMaxForm(strs, m, n):
    if not strs: return 0
    
    from collections import Counter
    strs_cntr = Counter(strs)
    costs = []
    for s, cnt in strs_cntr.items():
        n0 = 0
        for ch in s:
            if ch == '0': n0 += 1
        n1 = len(s) - n0
        costs += [(n0, n1, cnt)]

    tbl = {}
    def find_dp(idx, w0, w1):
        if idx < 0: return 0
        try: return tbl[(idx, w0, w1)]
        except: pass

        key = (idx, w0, w1)
        best = find_dp(idx - 1, w0, w1)
        c0, c1, reps = costs[idx]
        cnt = 0
        while w0 >= c0 and w1 >= c1 and cnt < reps:
            w0 -= c0; w1 -= c1; cnt += 1
            best_sans = cnt + find_dp(idx - 1, w0, w1)
            best = max(best, best_sans)
            
        tbl[key] = best
        return best

    return find_dp(len(costs) - 1, m, n)


def TEST(strs, m, n):
    print(findMaxForm(strs, m, n))

TEST(["10", "0001", "111001", "1", "0"], 5, 3)
TEST(["10", "0", "1"], 1, 1)

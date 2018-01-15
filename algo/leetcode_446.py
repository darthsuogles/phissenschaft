""" Arithmetic subsequences
"""

def numberOfArithmeticSlices(A):
    n = len(A)
    if n < 3: return 0
    if 1 == len(set(A)):
        cnts = 1 << n
        cnts -= (n - 1) * n // 2 + n + 1
        return cnts

    from collections import defaultdict
    pool = [defaultdict(int) for i in range(n)]
    cnts = 0
    for i, a in enumerate(A):
        rec = pool[i]
        for j, b in enumerate(A[:i]):
            step = a - b
            rec[step] += 1
            prev = pool[j][step]
            rec[step] += prev
            cnts += prev
                
    return cnts

def TEST(A):
    print("-----------")
    cnts = numberOfArithmeticSlices(A)
    print(cnts)

print("------TEST-CASES------")
TEST([2, 4, 6, 8, 10])
TEST([0, 1, 2, 2, 2])
TEST([2,2,3,4])
TEST([1,1,1,1,1])

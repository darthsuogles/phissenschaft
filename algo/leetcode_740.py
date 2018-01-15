"""
Delete and earn
"""

def deleteAndEarn(nums):
    if not nums: return 0
    from collections import Counter
    cntr = Counter(nums)
    arr = sorted(cntr.items())
    # Add a sentinel
    arr.append((arr[-1][0] + 2, 0))

    def search(idx, bnd):
        if idx >= bnd: return 0
        vals_tbl = [0] * (bnd - idx + 2)
        for i in range(0, bnd - idx):
            val, cnt = arr[i + idx]
            taken = val * cnt + vals_tbl[i-2]
            skip = vals_tbl[i-1]
            vals_tbl[i] = max(taken, skip)
        return vals_tbl[-3]

    cum_sum = 0
    idx = 0
    while idx + 1 < len(arr):
        prev, _ = arr[idx]
        for bnd in range(idx + 1, len(arr)):
            curr, _ = arr[bnd]
            if prev + 1 < curr:
                break
            prev = curr

        cum_sum += search(idx, bnd)
        idx = bnd

    return cum_sum

print(deleteAndEarn([2,2,3,3,3,4]))
print(deleteAndEarn([3,4,2]))
print(deleteAndEarn([3,1]))

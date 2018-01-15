"""
Find K closest elements
"""

def findClosestElements(arr, k, x):
    if not arr or k <= 0: return []
    from bisect import bisect_left
    idx = bisect_left(arr, x)
    cnts = 0
    N = len(arr)
    i, j = idx - 1, idx  # the actual insert position
    print(arr[i], arr[j])
    while cnts < k and i >= 0 and j < N:
        a, b = arr[i], arr[j]
        cnts += 1
        if x - a <= b - x:  # strict left preference
            i -= 1
        else:
            j += 1

    if cnts < k:
        if i == -1:
            j += (k - cnts)
            j = min(j, N)
        elif j == N:
            i -= (k - cnts)
            i = max(i, -1)
        else:
            assert "???"

    return arr[(i + 1) : j]

def TEST(arr, k, x):
    print('-----TEST-------', arr, k, x)
    res = findClosestElements(arr, k, x)
    print(res)

TEST([-1,0,1,2,3,4,5], 4, 3)
TEST([1,2,3,4,5], 4, -1)
TEST([0,0,1,2,3,3,4,7,7,8], 3, 5)

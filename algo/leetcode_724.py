''' H-index
'''

def hIndexIter(citations):
    arr = sorted(citations, reverse=True)
    h = 0
    for c in arr:
        if c < 1 + h: break
        h += 1
        
    return h
            

def hIndex(citations):
    if [] == citations: return 0
    arr = sorted(citations)
    n = len(arr)
    i = 0; j = n - 1
    while i + 1 < j:
        k = (i + j) // 2
        a = arr[k] - (n - k)
        if a == 0:
            return (n - k)
        if a < 0:
            i = k + 1
        else:
            j = k - 1

    # i is always a safe lower bound
    while i < n:
        if arr[i] >= (n - i):
            return n - i
        i += 1
    return 0


def TEST(arr, h):
    res = hIndex(arr)
    if res != h:
        print("Error", res, 'but expect', h)
    else:
        print("Ok")


TEST([3,0,6,1,5], 3)
TEST([], 0)
TEST([0], 0)
TEST([1,4,7,9], 3)
TEST([100], 1)
TEST([11, 15], 2)

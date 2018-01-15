''' Generate all permuateions
'''

from bisect import bisect_left

def all_perms_rec(arr):
    if not arr: return [[]]
    res = []
    _seen = set()  # record duplicates
    for i, a in enumerate(arr):
        if a in _seen: continue
        _seen.add(a)
        arr[0], arr[i] = a, arr[0]
        sub_perms = all_perms_rec(arr[1:])
        arr[0], arr[i] = arr[i], a
        for _perm in sub_perms:
            res.append([a] + _perm)

    return res


def next_perm(arr):
    ''' With inplace modification
    '''
    # No need for empty or single list
    if not arr or 0 == len(arr): 
        return
    
    # Check inversely sorted postfix
    # These are the highest order that one cannot alter
    n = len(arr)
    i = n - 1
    while i > 0 and arr[i-1] >= arr[i]:
        i -= 1

    # The postfix a[i] .. a[n-1] is not the full sequence
    # With, "a[i-1] < a[i] >= .." find the first a[j] > a[i-1]
    # In this case, there is at least one such i <= j < n
    if i > 0:
        j = n - 1
        tgt = arr[i-1]
        while j >= i and tgt >= arr[j]:
            j -= 1

        arr[j], arr[i-1] = arr[i-1], arr[j]

    # Revert the postfix
    k = i; j = n - 1
    while k < j:
        arr[k], arr[j] = arr[j], arr[k]
        k += 1; j -= 1

    print(arr)

next_perm([5, 1, 1])

# arr = [1,2,3]
# for i in range(10):
#     next_perm(arr)

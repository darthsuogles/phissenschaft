""" Largest distance
"""

def maxDistance(arrays):
    if not arrays: return 0
    n = len(arrays)

    _min, _max = arrays[0][0], arrays[0][-1]
    excl_min = [None] * n
    excl_max = [None] * n
    for i in range(1, n):
        elems = arrays[i]
        excl_min[i] = _min
        excl_max[i] = _max
        _min = min(_min, elems[0])
        _max = max(_max, elems[-1])

    max_diff = 0
    _min, _max = arrays[-1][0], arrays[-1][-1]
    for i in range(n-2, 0, -1):
        p, q = arrays[i][0], arrays[i][-1]
        curr = max(q - min(_min, excl_min[i]), 
                   max(_max, excl_max[i]) - p)
        max_diff = max(curr, max_diff)
        _min = min(_min, p)
        _max = max(_max, q)
        
    curr = max(arrays[0][-1] - _min,
               _max - arrays[0][0])
    max_diff = max(curr, max_diff)

    return max_diff


print(maxDistance([[1,2,3],
                   [4,5],
                   [1,2,3]]))

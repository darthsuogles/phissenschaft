''' Search in a row/column sorted matrix
'''

def bisect_left(nums, a):
    if nums is None: return None
    n = len(nums)
    if 0 == n: return 0
    
    i = 0; j = n-1
    while i + 1 < j:
        k = (i + j) // 2
        v = nums[k]
        if a == v: 
            while k >= 0:
                if nums[k] != a:
                    break
                k -= 1
            return k + 1
        if a < v:
            j = k
        else:
            i = k

    if a <= nums[i]:
        return i
    elif nums[j] < a:
        return j+1
    return j
            

# Return any match index, or None if no match
def find_in_array(nums, a):
    if nums is None: return None
    n = len(nums)
    if 0 == n: return 0
    
    i = 0; j = n-1
    while i + 1 < j:
        k = (i + j) // 2
        v = nums[k]
        if a == v: 
            return k
        if a < v:
            j = k
        else:
            i = k

    if a == nums[i]:
        return i
    elif a == nums[j]:
        return j
    return None
    

# Assuming all rows are non-empty
def search_matrix(X, a):
    if X is None: return None
    m = len(X)
    if 0 == m: return None
    
    i = 0; j = m-1
    while i + 1 < j:
        k = (i + j) // 2
        curr = X[k][0]
        if curr == a:
            return (i, 0)
        if curr < a:
            i = k
        else:
            j = k

    if a == X[i][0]:
        return (i, 0)
    if a == X[j][0]:
        return (j, 0)

    def find_in_slice(i, a):
        res = find_in_array(X[i], a)
        if res is None: return None
        return (i, res)

    if a < X[i][0]:
        if 0 == i:
            return None
        return find_in_slice(i-1, a)

    if X[j][0] < a:
        if X[j][-1] < a:
            return None
        return find_in_slice(j, a)

    return find_in_slice(i, a)


# Deal with empty rows
def find_in_matrix(X, a):
    Xt = []
    inds = []
    for i, row in enumerate(X):
        if row is None or [] == row:
            continue
        inds += [i]
        Xt += [row]

    res = search_matrix(Xt, a)
    if res is None: return None
    i, j = res
    return (inds[i], j)


def TEST(X, a):
    res = find_in_matrix(X, a)
    if res is None:
        print('val', a, 'not found')
        return
    i, j = res
    if a != X[i][j]:
        print('Error', X[i][j], 'but expect', a)
    else:
        print('found val', X[i][j], 'at', i, j)


TEST([[1,2,3], 
      [5,7], 
      [],
      [9,11,16]], 16)

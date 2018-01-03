""" Find nearest zero in matrix
"""

def updateMatrix(matrix):
    m = len(matrix)
    if 0 == m: return [[]]
    n = len(matrix[0])
    if 0 == n: return [] * m
    
    tbl = []
    for i, row in enumerate(matrix):
        tbl_row = [(m + n)] * n
        for j, bit in enumerate(row):            
            if 0 == bit:
                tbl_row[j] = 0
        tbl.append(tbl_row)

    # Forward pass
    for i in range(m):
        for j in range(n):
            if 0 == tbl[i][j]: continue
            min_dist = tbl[i][j]
            if i > 0:
                min_dist = min(min_dist, 1 + tbl[i-1][j])
            if j > 0:
                min_dist = min(min_dist, 1 + tbl[i][j-1])
            tbl[i][j] = min_dist

    # Backward pass: this actually works!
    for i in range(m-1, -1, -1):
        for j in range(n-1, -1, -1):
            if 0 == tbl[i][j]: continue
            min_dist = tbl[i][j]
            if i + 1 < m:
                min_dist = min(min_dist, 1 + tbl[i+1][j])
            if j + 1 < n:
                min_dist = min(min_dist, 1 + tbl[i][j+1])
            tbl[i][j] = min_dist
            
    return tbl


def TEST(A, tgt=None):
    res = updateMatrix(A)

    if not tgt:
        for row in res:
            print(row)
        print('--------------')
        return

    for row_res, row_tgt in zip(res, tgt):
        if row_res == row_tgt:
            print(row_res)
        else:
            print('ERROR:init!')
            print(row_res); print(row_tgt)
            print('ERROR:fini!')
    print('--------------')


print('-------TEST CASES---------')
TEST([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])

TEST([
    [0, 0, 0],
    [0, 1, 0],
    [1, 1, 1]
])


ref = [
    [1,0,1,1,0,0,1,0,0,1],
    [0,1,1,0,1,0,1,0,1,1],
    [0,0,1,0,1,0,0,1,0,0],
    [1,0,1,0,1,1,1,1,1,1],
    [0,1,0,1,1,0,0,0,0,1],
    [0,0,1,0,1,1,1,0,1,0],
    [0,1,0,1,0,1,0,0,1,1],
    [1,0,0,0,1,2,1,1,0,1],[2,1,1,1,1,2,1,0,1,0],[3,2,2,1,0,1,0,0,1,1]]

TEST([
    [1, 0, 1, 1, 0, 0, 1, 0, 0, 1], 
    [0, 1, 1, 0, 1, 0, 1, 0, 1, 1], 
    [0, 0, 1, 0, 1, 0, 0, 1, 0, 0], 
    [1, 0, 1, 0, 1, 1, 1, 1, 1, 1], 
    [0, 1, 0, 1, 1, 0, 0, 0, 0, 1], 
    [0, 0, 1, 0, 1, 1, 1, 0, 1, 0], 
    [0, 1, 0, 1, 0, 1, 0, 0, 1, 1], 
    [1, 0, 0, 0, 1, 1, 1, 1, 0, 1], 
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 0], 
    [1, 1, 1, 1, 0, 1, 0, 0, 1, 1]], ref)

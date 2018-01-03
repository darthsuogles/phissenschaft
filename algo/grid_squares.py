''' Find number of squares in a rectangular grid
'''

def find_num_squares(m, n):
    if 0 == m:
        if 0 != n: return n
    if 0 == n:
        return m

    tbl = []  # with ghost cells
    for i in range(m+1):
        tbl.append([0] * (n + 1))

    for i in range(1, m+1):
        for j in range(1, n+1):
            prev = tbl[i][j-1] + tbl[i-1][j] - tbl[i-1][j-1]
            tbl[i][j] = min(i, j) + prev
            
    return tbl[m][n]


print(find_num_squares(2, 2))

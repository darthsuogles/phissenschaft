''' Min path sum in a matrix
'''

def minPathSum(grid):
    if not grid: return -1
    m = len(grid)
    n = len(grid[0])
    if 0 == n: return -1

    path_sum = []
    for i in range(m):
        path_sum.append([None] * n)

    path_sum[0][0] = grid[0][0]
    for j in range(1, n):
        path_sum[0][j] = path_sum[0][j-1] + grid[0][j]
    for i in range(1, m):
        path_sum[i][0] = path_sum[i-1][0] + grid[i][0]

    for i in range(1, m):
        for j in range(1, n):
            prev = min(path_sum[i-1][j], path_sum[i][j-1])
            path_sum[i][j] = prev + grid[i][j]

    return path_sum[m-1][n-1]



    

''' Compute all magic squares
''' 

def get_all_magic_squares(n):
    if n < 1: return []
    if 0 != (n + 1) % 2: return []    
    pivot = (n * n + 1) // 2  # row / col / diags sum
    
    grid = []
    for i in range(n):
        grid.append([None] * n)
    
    i = n // 2  # middle element
    grid[i][i] = pivot
    common_sum = pivot * n

    def validate(grid, s, t):
        if s < 0 or s >= n or t < 0 or t >= n: 
            return False

        col_sum = 0
        for i in range(n):
            if grid[i][t] is None: continue
            col_sum += grid[i][t]
            if col_sum > common_sum: return False
            
        row_sum = 0
        for j in range(n):
            if grid[s][j] is None: continue
            row_sum += grid[s][j]
            if row_sum > common_sum: return False
                
        if s == t:
            diag_sum = 0
            for i in range(n):
                if grid[i][i] is None: continue
                diag_sum += grid[i][i]
                if diag_sum > common_sum: return False

        if s == n - t - 1:
            diag_sum = 0        
            for i in range(n):
                if grid[i][n-i-1] is None: continue
                diag_sum += grid[i][n-i-1]
                if diag_sum > common_sum: return False

        return True

    def dfs(grid, avail_nums):
        if not avail_nums:
            print('---------------------------')
            for row in grid:
                print(' | '.join(map(str, row)))
            return

        a = avail_nums[0]
        avail_sub = avail_nums[1:]
        for i in range(n):
            for j in range(n):
                if grid[i][j] is not None:
                    continue
                grid[i][j] = a
                if validate(grid, i, j):                       
                    dfs(grid, avail_sub)
                grid[i][j] = None


    n2 = n * n
    avail_nums = [k for k in range(1, n2 + 1) if k != pivot]
    print(avail_nums)
    dfs(grid, avail_nums)
    

get_all_magic_squares(3)

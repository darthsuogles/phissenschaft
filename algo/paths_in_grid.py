''' Number of paths from a certain grid
'''

def count_paths(grid, i, j, d, path):
    if -1 == grid[i][j]: 
        return 0

    m = len(grid)
    n = len(grid[0])

    if 0 == d:        
        print(path)
        canvas = []
        for s in range(1+m): 
            canvas += [[' '] * (2 * n)]
            canvas += [[' '] * (2 * n)]

        _plt = [['\\', '|', '/'], 
                ['-', '#', '-'], 
                ['/',  '|', '\\']]

        for p, q in zip(path[:-1], path[1:]):
            di = p[0] - q[0]; dj = p[1] - q[1]
            ci = 2 * q[0]; cj = 2 * q[1]
            canvas[ci][cj] = '*'
            ch_prev = canvas[ci + di][cj + dj]
            if ' ' == ch_prev: 
                ch = _plt[1+di][1+dj]
            elif ch_prev in ['-', '|']:
                ch = '+'
            else:
                ch = 'X'
            canvas[ci + di][cj + dj] = ch
        
        canvas[2 * path[0][0]][2 * path[0][1]] = '@'
        for row in canvas:
            for ch in row:
                print(ch + ' ', end='')
            print()
        return 1


    grid[i][j] = -1
    res = 0
    for s in range(max(0, i-1), 
                   min(m-1, i+1) + 1):
        for t in range(max(0, j-1),
                       min(n-1, j+1) + 1):
            if (s != i) or (t != j):
                res += count_paths(grid, s, t, d - 1, path + [(s, t)])

    grid[i][j] = 0
    return res


def count_all_paths(m, n, i, j, d):
    grid = []
    for s in range(m):
        grid += [[0] * n]
    return count_paths(grid, i, j, d, [(i, j)])


def TEST(m, n, i, j, d):
    print(count_all_paths(m, n, i, j, d))


TEST(3, 3, 1, 1, 4)

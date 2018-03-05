"""
Swim in water in a grid
"""

# TODO: https://leetcode.com/problems/swim-in-rising-water/solution/
def swimInWater(grid):
    n = len(grid)
    if 0 == n: return 0
    assert len(grid[0]) == n

    _MAX_VAL = n * n
    tbl_fwd = []
    for _ in range(n + 2):
        tbl_fwd.append([_MAX_VAL] * (n + 2))

    # Bellman Ford shortest path algorithm
    # But Dijastra should work just fine.
    tbl_fwd[0][1] = tbl_fwd[1][0] = 0
    for step in range(n * n):
        num_updates = 0
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                curr = _MAX_VAL
                curr = min(curr, tbl_fwd[i-1][j])
                curr = min(curr, tbl_fwd[i+1][j])
                curr = min(curr, tbl_fwd[i][j-1])
                curr = min(curr, tbl_fwd[i][j+1])
                curr = max(curr, grid[i-1][j-1])
                if tbl_fwd[i][j] != curr:
                    num_updates += 1
                    tbl_fwd[i][j] = curr

        if 0 == num_updates:
            break

    return tbl_fwd[n][n]


def TEST(grid):
    print(swimInWater(grid))


TEST([[0,2],[1,3]])
TEST([[0,1,2,3,4],
      [24,23,22,21,5],
      [12,13,14,15,16],
      [11,17,18,19,20],
      [10,9,8,7,6]])

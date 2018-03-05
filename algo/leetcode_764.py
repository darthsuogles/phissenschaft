"""
Largest plus sign in a mined grid
"""

def orderOfLargestPlusSign(N, mines):
    if N < 1: return 0
    if len(mines) == N * N: return 0

    banned = set(tuple(ij) for ij in mines)
    tbl = []
    for _ in range(N): tbl.append([0] * N)

    for i in range(N):
        cnt = 0
        for k in range(0, N):
            cnt = 0 if (i, k) in banned else cnt + 1
            tbl[i][k] = cnt

        cnt = 0
        for k in range(N-1, -1, -1):
            cnt = 0 if (i, k) in banned else cnt + 1
            tbl[i][k] = min(tbl[i][k], cnt)

    for j in range(N):
        cnt = 0
        for k in range(0, N):
            cnt = 0 if (k, j) in banned else cnt + 1
            tbl[k][j] = min(tbl[k][j], cnt)

        cnt = 0
        for k in range(N-1, -1, -1):
            cnt = 0 if (k, j) in banned else cnt + 1
            tbl[k][j] = min(tbl[k][j], cnt)

    max_val = 0
    for i in range(N):
        for j in range(N):
            max_val = max(tbl[i][j], max_val)
    return max_val


def orderOfLargestPlusSignBruteForce(N, mines):
    if N < 1: return 0
    if len(mines) == N * N: return 0

    grid = []
    for _ in range(N): grid.append([True] * N)

    def check_diam_four(i, j, diam):
        d = diam - 1
        if i < d or i + d >= N or j < d or j + d >= N:
            return False
        if grid[i - d][j] is None: return False
        if grid[i + d][j] is None: return False
        if grid[i][j - d] is None: return False
        if grid[i][j + d] is None: return False
        return True

    for i, j in mines: grid[i][j] = None

    diam = 1
    while True:
        cnts = 0
        for i in range(N):
            for j in range(N):
                if not grid[i][j]:
                    continue
                if check_diam_four(i, j, diam + 1):
                    cnts += 1
                else:
                    grid[i][j] = False

        if 0 == cnts:
            return diam
        diam += 1

    return None


def TEST(N, mines):
    print(orderOfLargestPlusSign(N, mines))


TEST(5, [[4, 2]])
TEST(2, [])
TEST(1, [[0, 0]])
TEST(2, [[0, 1], [1, 0], [1, 1]])

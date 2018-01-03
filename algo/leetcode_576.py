"""
Out of boundary paths
"""

def findePaths(m, n, N, i, j):
    # Counts the total moves at each step
    def gen_new_tbl():
        return [[0] * n for _ in range(m)]
    curr_tbl = gen_new_tbl()
    next_tbl = gen_new_tbl()

    MOD = int(1e9 + 7)
    curr_tbl[i][j] = 1
    num_outs = 0
    for step in range(N):
        step_outs = 0
        for i in range(m):
            for j in range(n):
                # Check if the point is reachable
                curr_mult = curr_tbl[i][j]
                curr_outs = 0
                if 0 == curr_mult:
                    continue
                # Check if we can move
                if 0 == i:
                    curr_outs += curr_mult
                else:
                    next_tbl[i-1][j] += curr_mult
                if i + 1 == m:
                    curr_outs += curr_mult
                else:
                    next_tbl[i+1][j] += curr_mult
                if 0 == j:
                    curr_outs += curr_mult
                else:
                    next_tbl[i][j-1] += curr_mult
                if j + 1 == n:
                    curr_outs += curr_mult
                else:
                    next_tbl[i][j+1] += curr_mult

                step_outs = (step_outs + curr_outs) % MOD

        num_outs = (num_outs + step_outs) % MOD
        # print('>> step', step, '#outs', step_outs)
        # for r in range(m):
        #     print(' '.join(map(str, curr_tbl[r])))
        # print('----')

        curr_tbl = next_tbl
        next_tbl = gen_new_tbl()

    return int(num_outs)


def TEST(m, n, N, i, j):
    print('--------TEST--------')
    print(findePaths(m, n, N, i, j))


TEST(2, 2, 2, 0, 0)
TEST(1, 3, 3, 0, 1)
TEST(2, 3, 8, 1, 0)

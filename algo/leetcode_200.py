"""
Number of islands
"""

def numIslands(grid):
    """ Using Union-Find """
    m = len(grid)
    if not m: return 0
    n = len(grid[0])
    if not n: return 0

    parent = list(range(m * n))
    rank = [0] * (m * n)

    # Path compression
    def find_parent(u):
        p = parent[u]
        if p != parent[p]:
            p = find_parent(p)
            parent[u] = p
        return p

    def make_union(u, v):
        up = find_parent(u)
        up_rank = rank[up]
        vp = find_parent(v)
        vp_rank = rank[vp]
        if up_rank < vp_rank:
            parent[up] = vp
        else:
            parent[vp] = up
        if up_rank == vp_rank:
            rank[up] = up_rank + 1

    # Add all the edges and connect if possible
    for i in range(m):
        for j in range(n):
            if '1' != grid[i][j]:
                continue
            vid = i * n + j
            if j + 1 < n and '1' == grid[i][j+1]:
                make_union(vid, vid + 1)
            if i + 1 < m and '1' == grid[i+1][j]:
                make_union(vid, vid + n)

    # Find parents for all vertices
    island_inds = set()
    for i in range(m):
        for j in range(n):
            vid = i * n + j
            pid = find_parent(vid)
            x, y = pid // n, pid % n
            if '1' == grid[x][y]:
                island_inds.add((x, y))

    return len(island_inds)


def numIslandsDFS(grid):
    """ Using DFS """
    m = len(grid)
    if not m: return 0
    n = len(grid[0])
    if not n: return 0

    visited = set()
    def search(i, j):
        if i < 0 or i >= m or j < 0 or j >= n:
            return
        if (i, j) in visited or '0' == grid[i][j]:
            return
        visited.add((i, j))
        search(i-1, j)
        search(i+1, j)
        search(i, j-1)
        search(i, j+1)

    num_islands = 0
    for i in range(m):
        for j in range(n):
            if (i, j) in visited or '0' == grid[i][j]:
                continue
            num_islands += 1
            search(i, j)

    return num_islands


def TEST(grid):
    for row in grid:
        row = list(row)
    print(numIslands(grid))


TEST(['11000',
      '11000',
      '00100',
      '00011'])


TEST(["1", "1"])

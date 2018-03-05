"""
Remove one edge to form a tree
"""

def findRedundantConnection(edges):
    if not edges: return []
    from collections import defaultdict

    parent = defaultdict(int)
    rank = defaultdict(int)
    for u, v in edges:
        parent[u] = u
        parent[v] = v
        rank[u] = 0
        rank[v] = 0

    def find_parent(u):
        pu = parent[u]
        if pu != parent[pu]:
            pu = find_parent(pu)
        parent[u] = pu
        return pu

    def merge(u, v):
        pu = find_parent(u)
        pv = find_parent(v)
        if pu == pv:
            return False

        pu_rank = rank[pu]
        pv_rank = rank[pv]
        if pu_rank < pv_rank:
            parent[pu] = pv
        elif pv_rank < pu_rank:
            parent[pv] = pu
        else:
            parent[pu] = pv
            rank[pv] += 1
        return True

    for u, v in edges:
        if not merge(u, v):
            return [u, v]

    return []

def TEST(edges):
    print(findRedundantConnection(edges))


TEST([[1,2], [1,3], [2,3]])
TEST([[1,2], [2,3], [3,4], [1,4], [1,5]])

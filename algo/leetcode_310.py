
def findMinHeightTrees(n, edges):
    if 0 == n: 
        return []
    if 0 == len(edges):
        return range(n)

    from collections import defaultdict
    adj = defaultdict(lambda: [])
    for (u, v) in edges:
        adj[u] += [v]
        adj[v] += [u]
    
    verts = [u for (u, _) in 
             sorted(adj.items(), 
                    key=lambda el: len(el[1]),
                    reverse=True)]

    _cache = {}

    def find_max_depth_lobnd(root, u, g_min_depth):
        try: return _cache[(root, u)]
        except: pass

        max_depth = 1
        for v in adj[u]:
            if v != root:
                max_depth = max(max_depth,
                                1 + find_max_depth_lobnd(u, v, g_min_depth))
                if max_depth > g_min_depth:
                    break
                    
        _cache[(root, u)] = max_depth
        return max_depth
            

    g_min_depth = n
    nds = []
    for root in verts:
        max_depth = 0
        for u in adj[root]:
            curr_depth = find_max_depth_lobnd(root, u, g_min_depth)
            max_depth = max(max_depth, curr_depth)                
            if max_depth > g_min_depth:
                break

        if max_depth > g_min_depth:
            continue
        nds += [(root, max_depth)]
        g_min_depth = max_depth

    return [u for (u, d) in nds if d == g_min_depth]


def TEST(n, edges, tgt):
    res = findMinHeightTrees(n, edges)
    res = sorted(res)
    tgt = sorted(tgt)
    if res == tgt:
        print("OK")
    else:
        print("ERROR", res, "!=", tgt)


TEST(4, [[1, 0], [1, 2], [1, 3]], [1])
TEST(6, [[0, 3], [1, 3], [2, 3], [4, 3], [5, 4]], [3, 4])

''' Division eval
'''

def calcEquation(equations, values, queries):
    if len(equations) != len(values): return []
    from collections import defaultdict
    
    neighbors = defaultdict(list)
    for (a, b), val in zip(equations, values):
        neighbors[a].append((b, val))
        neighbors[b].append((a, 1.0 / val))
        
    def dfs(u, v, curr_prod, visited):
        if u not in neighbors: return None
        if u == v: return curr_prod

        for node, val in neighbors[u]:
            if node in visited: continue
            visited.add(node)
            res = dfs(node, v, val * curr_prod, visited)
            if res is not None:
                return res

        return None

    res = []
    for a, b in queries:
        res.append(dfs(a, b, 1.0, set()) or -1.0)
        
    return res


def calcEquationAllPairs(equations, values, queries):
    ''' Compute all pairs shortest path first
    '''
    if len(equations) != len(values): return []
    if not equations: return []

    from collections import defaultdict
    edges = {}
    symbols = set()
    for (a, b), val in zip(equations, values):
        edges[(a, b)] = val
        edges[(b, a)] = 1.0 / val
        symbols.add(a)
        symbols.add(b)

    symbols = list(symbols)
    for sym in symbols:
        new_edges = {}
        for i, a in enumerate(symbols[:-1]):
            for b in symbols[(i+1):]:
                try: 
                    new_edges[(a, b)] = edges[(a, b)]
                    new_edges[(b, a)] = edges[(b, a)]
                except: pass

                try: 
                    val = edges[(a, sym)] * edges[(sym, b)]
                    new_edges[(a, b)] = val
                    new_edges[(b, a)] = 1.0 / val
                except: pass

        edges = new_edges

    res = []
    symbols = set(symbols)
    for a, b in queries:
        if a == b:
            if a in symbols and a != 0.0:
                res.append(1.0)
            else:
                res.append(-1.0)
            continue

        try: res.append(edges[(a, b)])
        except: res.append(-1.0)

    return res


def TEST(equations, values, queries, tgt):
    res = calcEquationAllPairs(equations, values, queries)
    if res != tgt:
        print("ERROR", res, 'but expecting', tgt)
    else:
        print("OK")


TEST([['a', 'b'], ['b', 'c']],
     [2.0, 3.0],
     [["a", "c"], ["b", "a"], ["a", "e"], ["a", "a"], ["x", "x"]],
     [6.0, 0.5, -1.0, 1.0, -1.0])

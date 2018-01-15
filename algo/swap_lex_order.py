''' Given a string and an array of pairs indicating 
    swappable chars, find the lexicographically largest string
'''

def swapLexOrderUsedOnce(text, pairs):
    ''' This version, you can only use each operation once
    '''
    if not pairs: return text
    i, j = pairs[0]
    _used = set([i, j])
    rest_pairs = [[s, t] for s, t in pairs
                  if s not in _used and t not in _used]    
    
    _chars = list(text)
    _chars[i-1], _chars[j-1] = _chars[j-1], _chars[i-1]
    text_with = swapLexOrderUsedOnce(''.join(_chars), rest_pairs)
    _chars[i-1], _chars[j-1] = _chars[j-1], _chars[i-1]
    text_sans = swapLexOrderUsedOnce(''.join(_chars), pairs[1:])
    return max(text_with, text_sans)


def swapLexOrderSimpleBFS(text, pairs):
    ''' This version search on the text
    '''
    if not pairs or not text: return text
    queue = [text]
    visited = set([text])
    text_max = text
    while queue:        
        curr = queue[0]; queue = queue[1:]
        for i, j in pairs:            
            i -= 1; j -= 1
            if i > j: i, j = j, i
            _next = curr[:i] + curr[j] + curr[(i+1):j] + curr[i] + curr[(j+1):]
            if _next not in visited:
                text_max = max(text_max, _next)
                visited.add(_next)
                queue.append(_next)

    return text_max


def swapLexOrder(text, pairs):
    if not text or not pairs: return text
    # Preprocess pairs to get all transitive closures
    # Any connected components induces all pairs
    n = len(text)
    from collections import defaultdict

    # Using Union-Find structure
    _id = list(range(n))
    _rank = [1] * n

    def find(u):
        # With path compression
        p = _id[u]
        while p != _id[p]: 
            p = _id[p]
        # Everything on the path to root must be changed
        while u != p:
            tmp = _id[u]; _id[u] = p; u = tmp
        return p

    def union(u, v):
        # With union-by-rank
        pu = find(u); pv = find(v)
        if pu == pv: return
        ru, rv = _rank[pu], _rank[pv]
        if ru < rv:
            _id[u] = pv
        elif rv < ru:
            _id[v] = pu
        else:
            _id[v] = pu
            _rank[pu] += 1

    # Create connected components
    for i, j in pairs: union(i-1, j-1)
    
    pairs_cc = defaultdict(list)
    for u in range(n):
        pairs_cc[find(u)].append(u)

    assert sum(map(len, pairs_cc.values())) == n

    pairs_cc = [sorted(verts) 
                for verts in pairs_cc.values() 
                if len(verts) > 1]
            
    # Within a connected component, sort chars in descending order
    _chars = list(text)
    for inds in pairs_cc:
        cc_chars = sorted([text[i] for i in inds], reverse=True)
        for j, ch in zip(inds, cc_chars):
            _chars[j] = ch
    
    return ''.join(_chars)


def TEST(text, pairs, tgt):
    res = swapLexOrder(text, pairs)
    #res = swapLexOrderSimpleBFS(text, pairs)
    if res != tgt:
        print("Error: got", res, "but expect", tgt, sep='\n')
        print(res <= tgt)
    else:
        print("Ok")

print('-------- TEST CASES ----------')
TEST("abdc", [[1, 4], [3, 4]], "dbca")
TEST("lvvyfrbhgiyexoirhunnuejzhesylojwbyatfkrv", [
    [13,23], 
    [13,28], 
    [15,20], 
    [24,29], 
    [6,7], 
    [3,4], 
    [21,30], 
    [2,13], 
    [12,15], 
    [19,23], 
    [10,19], 
    [13,14], 
    [6,16], 
    [17,25], 
    [6,21], 
    [17,26], 
    [5,6], 
    [12,24]
], "lyyvurrhgxyzvonohunlfejihesiebjwbyatfkrv")

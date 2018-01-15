''' Find boomerang triplets
'''

def numberOfBoomerangs(points):
    if points is None: return 0
    n = len(points)
    if n < 3: return 0
    
    from collections import defaultdict
    res = 0
    
    for p in points:
        _tbl = defaultdict(lambda: 0)
        for q in points:        
            dx = p[0] - q[0]
            dy = p[1] - q[1]
            d = dx * dx + dy * dy
            _tbl[d] += 1
        for cnt in _tbl.values():
            res += cnt * (cnt - 1)

    return res


def TEST(points, tgt):
    res = numberOfBoomerangs(points)
    if res != tgt:
        print("Error", res, 'but expect', tgt)
    else:
        print("Ok")


TEST([[0, 0], [1, 0], [2, 0]], 2)

''' Max points on a line
'''

class Point(object):
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b


def maxPoints(points):
    if not points: return 0
    n = len(points)
    if n < 3: return n

    def gcd(a, b):
        if 0 == b: return a
        return gcd(b, a % b)

    def get_rational(px, py, qx, qy):
        x = px - qx; y = py - qy
        if 0 == x: return ((0, 1), (px, 0))
        if 0 == y: return ((1, 0), (0, py))
        if y < 0:  # to 1st or 2nd quadrant
            x, y = -x, -y

        d = gcd(abs(x), abs(y))
        s, t = (py * qx - px * qy), (px - qx)
        return ((x // d, y // d), (0, s / t))

    from collections import defaultdict    
    point_cntr = defaultdict(int)
    for p in points:
        point_cntr[(p.x, p.y)] += 1
    point_cnts = list(point_cntr.items())
    n = len(point_cnts)
    if 1 == n: return point_cnts[0][-1]
        
    max_cnt = 0
    line_tbl = {}
    for i, ((px, py), p_cnt) in enumerate(point_cnts[1:], 1):
        _seen = set()
        for (qx, qy), q_cnt in point_cnts[:i]:            
            key = get_rational(px, py, qx, qy)
            if key in _seen:
                continue
            _seen.add(key)
            if key not in line_tbl:
                val = q_cnt + p_cnt
            else:
                val = line_tbl[key] + p_cnt
            
            line_tbl[key] = val
            max_cnt = max(max_cnt, val)

    return max_cnt
                        
                      
def TEST(arr, tgt):
    points = [Point(p[0], p[1]) for p in arr]
    res = maxPoints(points)
    if res != tgt:
        print('Error', res, 'but expect', tgt)
    else:
        print('Ok')


TEST([], 0)
TEST([[0,0], [0,0]], 2)
TEST([[0,0], [1, 0], [1, 1], [2, 2]], 3)
TEST([[0,-1],[0,3],[0,-4],[0,-2],[0,-4],[0,0],[0,0],[0,1],[0,-2],[0,4]], 10)
TEST([[0,-12],[5,2],[2,5],[0,-5],[1,5],[2,-2],[5,-4],[3,4],[-2,4],[-1,4],[0,-5],[0,-8],[-2,-1],[0,-11],[0,-9]], 6)
TEST([[0,0], [1,0]], 2)
TEST([[0,0],[1,1],[1,-1]], 2)
TEST([[40,-23],[9,138],[429,115],[50,-17],[-3,80],[-10,33],[5,-21],[-3,80],[-6,-65],[-18,26],[-6,-65],[5,72],[0,77],[-9,86],[10,-2],[-8,85],[21,130],[18,-6],[-18,26],[-1,-15],[10,-2],[8,69],[-4,63],[0,3],[-4,40],[-7,84],[-8,7],[30,154],[16,-5],[6,90],[18,-6],[5,77],[-4,77],[7,-13],[-1,-45],[16,-5],[-9,86],[-16,11],[-7,84],[1,76],[3,77],[10,67],[1,-37],[-10,-81],[4,-11],[-20,13],[-10,77],[6,-17],[-27,2],[-10,-81],[10,-1],[-9,1],[-8,43],[2,2],[2,-21],[3,82],[8,-1],[10,-1],[-9,1],[-12,42],[16,-5],[-5,-61],[20,-7],[9,-35],[10,6],[12,106],[5,-21],[-5,82],[6,71],[-15,34],[-10,87],[-14,-12],[12,106],[-5,82],[-46,-45],[-4,63],[16,-5],[4,1],[-3,-53],[0,-17],[9,98],[-18,26],[-9,86],[2,77],[-2,-49],[1,76],[-3,-38],[-8,7],[-17,-37],[5,72],[10,-37],[-4,-57],[-3,-53],[3,74],[-3,-11],[-8,7],[1,88],[-12,42],[1,-37],[2,77],[-6,77],[5,72],[-4,-57],[-18,-33],[-12,42],[-9,86],[2,77],[-8,77],[-3,77],[9,-42],[16,41],[-29,-37],[0,-41],[-21,18],[-27,-34],[0,77],[3,74],[-7,-69],[-21,18],[27,146],[-20,13],[21,130],[-6,-65],[14,-4],[0,3],[9,-5],[6,-29],[-2,73],[-1,-15],[1,76],[-4,77],[6,-29]], 25)

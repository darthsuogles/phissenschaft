""" Jug problem
"""

def canMeasureWater(x, y, z):
    if z == x or z == y or x + y == z: return True
    if x == 0 or y == 0 or x + y < z: return False
    a, b = x, y
    while b != 0:
        a, b = b, a % b
    return 0 == (z % a)

def canMeasureWaterBF(x, y, z):
    if 0 == z: return True
    import heapq as hpq
    cands = [(0, 0)]
    visited = set()
    while cands:
        x_w, y_w = cands.pop(-1)

        # First the simple states
        next_states = [(x_w, 0),
                       (x_w, y),
                       (x, y_w),
                       (0, y_w)]
        # Then pouring water around
        x_s = x - x_w; y_s = y - y_w
        if x_s <= y_w:
            next_states += [(x, y_w - x_s)]
        else:            
            next_states += [(x_w + y_w, 0)]
        if y_s <= x_w:
            next_states += [(x_w - y_s, y)]
        else:
            next_states += [(0, y_w + x_w)]

        for a, b in next_states:
            if a == z or b == z or a + b == z:
                return True
            if (a, b) in visited:
                continue
            visited.add((a, b))
            cands.append((a, b))
        
    return False

def TEST(x, y, z):
    print(canMeasureWater(x, y, z))


TEST(0,0,0)
TEST(0,1,1)
TEST(0,0,1)
TEST(3,5,4)
TEST(2,6,5)
TEST(1,2,3)
TEST(104639, 104651, 234)
TEST(104659, 104677, 142424)

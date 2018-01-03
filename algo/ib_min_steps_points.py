'''
'''

def coverPoints(X, Y):
    if not X or not Y:
        return 0
    if len(X) == 1: 
        return 0
    
    vx = (abs(x0 - x1) for x0, x1 in zip(X[:-1], X[1:]))
    vy = (abs(y0 - y1) for y0, y1 in zip(Y[:-1], Y[1:]))
    return sum([max(x, y) for x, y in zip(vx, vy)])


def TEST(points):
    X, Y = zip(*points)
    print(coverPoints(X, Y))


TEST([(0, 0), (1, 1), (1, 2)])
TEST([(-7, 1), (-13, -5)])

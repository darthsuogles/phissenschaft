"""
Asteroid collision
"""

def asteroidCollision(asteroids):
    """
    Basically a matching parenthesis problem
    If however only the surviving velocity is of interest,
    then we can provide a constant space solution.
    """
    res = []
    for velo in asteroids:
        if velo >= 0:
            res.append(velo)
            continue
        while res:
            if res[-1] < 0:
                break
            anti = res[-1]
            if -velo < anti:
                velo = None; break
            res.pop()
            if -velo == anti:
                velo = None; break
        if velo is not None:
            res.append(velo)
    return res

def asteroidCollisionStack(asteroids):
    res = []
    for v in asteroids:
        if v > 0:
            res.append(v); continue
        retain = True
        while res:
            a = res[-1]
            if a < 0:
                break
            if a >= -v:
                if a == -v:
                    res.pop(-1)
                retain = False
                break
            res.pop(-1)

        if retain:
            res.append(v)

    return res

def TEST(asteroids):
    print(asteroidCollision(asteroids))

TEST([5, 10, -5])
TEST([8, -8])
TEST([10, 2, -5])
TEST([-2, -1, 1, 2])

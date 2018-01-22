"""
The skyline problem
"""

def getSkylineOne(buildings):
    if not buildings: return []
    # building: [left_pos, right_pos, height]
    from collections import namedtuple
    ChangePoint = namedtuple("ChangePoint", ['x', 'init', 'h'])

    change_pts = []
    for p0, p1, h in buildings:
        change_pts.append(ChangePoint(x=p0, init=True, h=h))
        change_pts.append(ChangePoint(x=p1, init=False, h=h))

    change_pts = sorted(change_pts, key=lambda pt: pt.x)

    # Store the overlay buildings
    from bisect import bisect_left
    hpq = []

    def overlay_adjust(pt):
        if pt.init:
            hpq.insert(bisect_left(hpq, pt.h), pt.h)
        else:
            hpq.remove(pt.h)

    def overlay_hmax():
        return 0 if not hpq else hpq[-1]

    area = 0
    res = []
    for curr in change_pts:
        overlay_adjust(curr)
        h_max = overlay_hmax()
        if h_max > curr.h:
            continue
        # Adjacent buildings
        while res and res[-1][0] == curr.x:
            res.pop(-1)
        # Mergeable points
        x = curr.x
        while res and res[-1][1] == h_max:
            x, _ = res.pop(-1)
        res.append([x, h_max])

    return res


def getSkyline(buildings):
    import heapq
    # heapq implements min-heap, thus need `negH`
    events = sorted([(L, -H, R) for L, R, H in buildings]
                    + list(set((R, 0, None) for _, R, _ in buildings)))

    # Add sentinels to simplify boundary condition checking
    res, hpq = [[0, 0]], [(0, float("inf"))]  # infinite horizon
    area = 0
    for x, negH, R in events:
        # Clear out-of-range overlay layers
        # This way we don't have to remove a particular item
        # It does not remove every out-of-range layer at once.
        # But we don't really care about this
        while x >= hpq[0][1]:
            heapq.heappop(hpq)
        if negH:
            heapq.heappush(hpq, (negH, R))

        # Check if mergeable: previous
        h_next = -hpq[0][0]
        x_prev, h_prev = res[-1]
        if h_next != h_prev:
            res += [x, h_next],
            area += (x - x_prev) * h_prev

    return res[1:]


def TEST(buildings):
    print(getSkyline(buildings))

TEST([[2, 9, 10],
      [3, 7, 15],
      [5, 12, 12],
      [15, 20, 10],
      [19, 24, 8]])

TEST([[0,2,3],[2,5,3]])

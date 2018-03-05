"""
The skyline problem
"""

def getSkylineArea(buildings):
    if not buildings: return []
    # building: [left_pos, right_pos, height]
    from collections import namedtuple
    ChangePoint = namedtuple("ChangePoint", ['x', 'init', 'neg_h'])

    change_pts = []
    for p0, p1, h in buildings:
        change_pts.append(ChangePoint(x=p0, init=0, neg_h=-h))
        change_pts.append(ChangePoint(x=p1, init=1, neg_h=-h))

    # If two change points have the same `x` location, we want
    # the one that is an initial point to go first. If both
    # are initial points, the one that's tallest should win.
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
    events = sorted([(x0, -y, x1) for x0, x1, y in buildings]
                    + list(set((x1, 0, None) for _, x1, _ in buildings)))

    # Add sentinels to simplify boundary condition checking
    res = [[0, 0]]
    hpq = [(0, float("inf"))]  # infinite horizon
    area = 0
    for x0, neg_y, x1 in events:
        # Clear out-of-range overlay layers
        # This way we don't have to remove a particular item
        # It does not remove every out-of-range layer at once.
        # But we don't really care about this
        while x0 >= hpq[0][1]:
            heapq.heappop(hpq)

        # Only add a new one if it has a real height.
        # The falling cliffs are annoted with height = 0.
        if neg_y < 0:
            heapq.heappush(hpq, (neg_y, x1))

        # See if we need a new line segment in the resulting skyline.
        # Either the current event is subsumed by the tallest at
        # the moment or that we may have adjacent buildings with
        # the same heights.
        y_curr = -hpq[0][0]
        x_prev, y_prev = res[-1]
        if y_curr != y_prev:
            res += [x0, y_curr],
            area += (x0 - x_prev) * y_curr

    return res[1:]


def TEST(buildings, tgt_pts):
    res_pts = getSkyline(buildings)
    assert res_pts == tgt_pts
    print('PASS')

TEST([[2, 9, 10], [3, 7, 15], [5, 12, 12], [15, 20, 10], [19, 24, 8]],
     [[2, 10], [3, 15], [7, 12], [12, 0], [15, 10], [20, 8], [24, 0]])

TEST([[0,2,3],[2,5,3]], [[0, 3], [5, 0]])

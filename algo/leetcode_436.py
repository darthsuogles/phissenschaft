""" Find minimum right interval
"""

class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e
        
    @classmethod
    def get(cls, ls):
        _vals = []
        for i, j in ls:
            _vals.append(Interval(i, j))
        return _vals


def findRightInterval(intervals):
    if not intervals: return []
    _end_pts = sorted([(intv.start, idx) for idx, intv in enumerate(intervals)])
    end_pts, end_inds = zip(*_end_pts)
    from bisect import bisect_left
    inds = []
    for intv in intervals:        
        k = bisect_left(end_pts, intv.end)
        if k == len(intervals):
            inds.append(-1)
        else:
            inds.append(end_inds[k])

    return inds


def TEST(intervals):
    print(findRightInterval(Interval.get(intervals)))

TEST([[1,2]])
TEST([[3,4], [2,3], [1,2]])
TEST([[1,4], [2,3], [3,4]])        
TEST([[4,5],[2,3],[1,2]])

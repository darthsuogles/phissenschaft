"""
Merge intervals
"""

def merge(intervals):
    if not intervals: return []
    sorted_intervals = sorted(intervals)
    result = []
    prev_intv = sorted_intervals[0]
    for curr_intv in sorted_intervals[1:]:
        if curr_intv[0] <= prev_intv[1]:
            prev_intv = (prev_intv[0], max(prev_intv[1], curr_intv[1]))
        else:
            result.append(prev_intv)
            prev_intv = curr_intv
    if prev_intv:
        result.append(prev_intv)

    return result


def TEST(intervals):
    print(merge(intervals))

TEST([[1,4], [2,3]])
TEST([[1,3],[2,6],[8,10],[15,18]])

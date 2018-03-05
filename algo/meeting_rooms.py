"""
Check if someone can attend all the meetings
"""

def check_non_overlapping(meetings):
    if not meetings: return True
    meetings = sorted(meetings, key=lambda intv: intv[1])
    n = len(meetings)
    # The return value i is such that all e in a[:i] have e <= x,
    # and all e in  a[i:] have e > x.
    from bisect import bisect_right
    end_times = [intv[1] for intv in meetings]
    for i in range(n - 1, -1, -1):
        intv = meetings[i]
        j = bisect_right(end_times, intv[0])
        if j < i: return False

    return True

def TEST(meetings):
    print(check_non_overlapping(meetings))

TEST([[1,2], [2,3], [3,4], [2,5]])

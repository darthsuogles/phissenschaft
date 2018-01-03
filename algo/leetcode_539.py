''' Minimum time difference
'''

def findMinDifference(timePoints):
    
    def time_to_int(tp_str):
        hh, mm = list(map(int, tp_str.split(':')))
        return hh * 60 + mm

    tps = sorted(map(time_to_int, timePoints))
    min_diff = min(t2 - t1 for t1, t2 in zip(tps[:-1], tps[1:]))
    min_diff = min(min_diff, (1440 - tps[-1] + tps[0]))
    return min_diff

res = findMinDifference(['23:59', '00:00'])
print(res)

''' Heater coverage 
'''

def findRadius(houses, heaters):
    if houses is None or 0 == len(houses):
        return -1
    if heaters is None or 0 == len(heaters):
        return -1

    from bisect import bisect_left
    
    heaters = sorted(heaters)
    max_radius = 0
    for haus in houses:
        i = bisect_left(heaters, haus)
        rad = None
        if i < len(heaters):
            rad = heaters[i] - haus
        if 0 < i:
            if rad is None:
                rad = haus - heaters[i-1]
            else:
                rad = min(rad, haus - heaters[i-1])

        max_radius = max(max_radius, rad)

    return max_radius


def TEST(houses, heaters, tgt_rad):
    res = findRadius(houses, heaters)
    if res != tgt_rad:
        print("ERROR", res, "but expecting", tgt_rad)
    else:
        print("OK")


TEST([1,2,3], [2], 1)
TEST([1,2,3,4], [1,4], 1)

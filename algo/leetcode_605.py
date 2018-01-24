"""
Can place flowers?
"""

def canPlaceFlowers(flowerbed, n):
    if not flowerbed: return 0 == n
    i = 0
    cnts = 0
    # The boundary is always non-competing
    flowerbed = [0] + flowerbed + [0]
    while i < len(flowerbed):
        if 1 == flowerbed[i]:
            i += 1; continue

        # Find consecutive non-planted slots
        j = i
        while j < len(flowerbed):
            if flowerbed[j] != 0:
                break
            j += 1

        cnts += (j - i - 1) // 2
        if cnts >= n: return True
        i = j

    return False

def TEST(flowerbed, n):
    print(canPlaceFlowers(flowerbed, n))


TEST([1,0,0,0,1], 1)
TEST([1,0,0,0,1], 2)
TEST([0,0,1,0,1], 1)

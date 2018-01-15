''' Consecutive haus
'''

def rob(nums):
    if nums is None: return 0
    
    tbl = [0, 0]
    for a in nums:
        curr = max(tbl[-1], a + tbl[-2])
        tbl = [tbl[-1], curr]

    return tbl[-1]


def TEST(nums):
    print(rob(nums))


TEST([5,2,3,4])

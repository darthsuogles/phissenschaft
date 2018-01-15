''' Perfect squares sum min nums
'''

def numSquaresBF(n, tbl={}):
    ''' This is relatively slow
    '''
    if 1 == n: return 1
    try: return tbl[n]
    except: pass
    sqrt = int(n ** 0.5)
    sq = sqrt * sqrt
    if sq == n:
        return 1

    res = n
    for k in range(sqrt, 0, -1):
        res = min(res, 1 + numSquaresBF(n - k*k, tbl))

    tbl[n] = res
    return res
    

def numSquares(n):
    ''' Using the coin change algorithm
    '''
    tbl = [None] * (1 + n)
    tbl[0] = 0
    n_sqrt = int(n ** 0.5)  # closest square root
    if n_sqrt * n_sqrt == n:
        return 1
    for i in range(1, n_sqrt + 1):
        a = i * i
        for j in range(n - a + 1):
            if tbl[j] is None: continue
            prev = tbl[j + a] or n
            tbl[j + a] = min(prev, tbl[j] + 1)

    return tbl[n]

        
def TEST(n):
    print(n, 'needs', numSquares(n))


TEST(12)
TEST(13)
TEST(16)
TEST(15)
TEST(5)
TEST(2)
TEST(9375)

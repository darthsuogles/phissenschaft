
def fourSumCountBF(A, B, C, D):

    from bisect import bisect_left

    tbl2 = {}
    tbl3 = {}
    C_s = sorted(C)
    D_s = sorted(D)

    def twoSumCount(s):
        try: return tbl2[s]
        except: pass
        cnt2 = 0
        i = 0
        _prev = C_s[0] - 1
        _prev_cnt = -1
        for c in C_s:   
            if c == _prev:
                cnt2 += _prev_cnt
                continue
            if s + c + D_s[0] > 0:
                break

            _prev = c
            _prev_cnt = 0
            j = bisect_left(D_s, -(s + c))
            while j < len(D_s):
                if s + c + D_s[j] > 0:
                    break
                _prev_cnt += 1
                j += 1
            cnt2 += _prev_cnt

        tbl2[s] = cnt2
        return cnt2


    B_s = sorted(B)
    def threeSumCount(a):
        try: return tbl3[a]
        except: pass
        cnt3 = 0
        b_bnd = a + C_s[0] + D_s[0]
        for b in B_s:
            if b + b_bnd > 0:
                break
            cnt3 += twoSumCount(a + b)
        tbl3[a] = cnt3
        return cnt3


    cnt4 = 0
    A_s = sorted(A)
    a_bnd = B_s[0] + C_s[0] + D_s[0]
    for a in A_s:
        if a + a_bnd > 0:
            break
        cnt4 += threeSumCount(a)    
    return cnt4


def fourSumCountBF2(A, B, C, D):
    C = sorted(C)
    D = sorted(D)

    tbl2 = {}
    from bisect import bisect_left

    def twoSumCount(s):
        try: return tbl2[s]
        except: pass
        cnt2 = 0
        i = 0
        _prev = C[0] - 1
        _prev_cnt = -1
        for c in C:   
            if c == _prev:
                cnt2 += _prev_cnt
                continue
            if s + c + D[0] > 0:
                break

            _prev = c
            _prev_cnt = 0
            j = bisect_left(D, -(s + c))
            while j < len(D):
                if s + c + D[j] > 0:
                    break
                _prev_cnt += 1
                j += 1
            cnt2 += _prev_cnt

        tbl2[s] = cnt2
        return cnt2
    
    # Main loop
    cnt = 0

    AB = sorted((a + b) for a in A for b in B)
    for s in AB:
        if s + C[0] + D[0] > 0:
            break
        cnt += twoSumCount(s)
        
    return cnt


def fourSumCount(A, B, C, D):
    from collections import defaultdict
    tbl = defaultdict(lambda: 0)
    for a in A:
        for b in B:
            tbl[a + b] += 1

    cnt = 0
    for c in C:
        for d in D:
            cnt += tbl[-(c + d)]
    return cnt


def TEST(A, B, C, D, tgt):
    res = fourSumCount(A, B, C, D)
    if res != tgt:
        print("Error {} != {}".format(res, tgt))
        print(tbl2, tbl3)
    else:
        print("OK")


TEST([1,2,1], [-2,-1], [-1,2], [0, 2], 3)
TEST([-1,1,1,1,-1],
     [0,-1,-1,0,1],
     [-1,-1,1,-1,-1],
     [0,1,0,-1,-1], 
     132)

""" All combinations of binary watch
"""

def readBinaryWatch(num):
    
    def show_all_binary(nbits, n):
        if nbits > n: return []
        if nbits == n: return [(1 << nbits) - 1]
        if 0 == nbits: return [0]
        res = []
        for a in show_all_binary(nbits, n-1):
            res.append(a << 1)
        for a in show_all_binary(nbits-1, n-1):
            res.append(1 + (a << 1))
        return res

    res = []
    for m in range(0, min(num, 4) + 1):
        hhs = show_all_binary(m, 4)
        mms = show_all_binary(max(0, min(6, num - m)), 6)
        for hh in hhs:
            if hh >= 12: continue
            for mm in mms:
                if mm >= 60: continue
                res.append("{:d}:{:02d}".format(hh, mm))
            
    return res


print(readBinaryWatch(1))

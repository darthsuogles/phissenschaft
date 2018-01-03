''' Add binary strings
'''

def addBinaryStrings(a, b):
    if not a: return b
    if not b: return a

    if len(a) < len(b): 
        a, b = b, a
    m, n = len(a), len(b)
    i = -1
    i_min = -m-1  # m is the larger
    carry = 0
    res = []
    for i in range(-1, i_min, -1):
        va = int(a[i])
        vb = 0 if i < -n else int(b[i])
        rem = va ^ vb ^ carry  # reminder of this position
        res.append(rem)
        carry = 1 if (va + vb + carry) > 1 else 0

    if carry > 0:
        res.append(carry)
        
    return ''.join(map(str, res[::-1]))


def TEST(a, b):
    res = addBinaryStrings(a, b)
    print(a, '+', b, '=', res)


TEST('1000', '111')
TEST('1', '1')

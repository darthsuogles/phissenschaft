''' Multiply two strings 
'''

def multiplyTwoStrings(s1, s2):
    if not s1: return s2
    if not s2: return s1
    
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    m = len(s1); n = len(s2)

    # Add two string based numbers
    def add_str(s1, s2):
        if not s1: return s2
        if not s2: return s1
        
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        m = len(s1); n = len(s2)

        carry = 0
        res = []
        for i in range(-1, -m-1, -1):
            a = int(s1[i])
            b = 0 if i < -n else int(s2[i])
            val = a + b + carry
            res.append(val % 10)
            carry = 1 if val >= 10 else 0

        if carry != 0:
            res.append(carry)

        return ''.join(map(str, res[::-1]))

    # Perform addition for each
    res_add = None
    for i in range(n):
        a = int(s2[n-i-1])
        carry = 0
        curr = ['0'] * i
        for j in range(m):
            b = int(s1[m-j-1])
            val = a * b + carry
            curr.append(val % 10)
            carry = val // 10
        if carry > 0:
            curr.append(carry)

        curr = ''.join(map(str, curr[::-1]))
        if res_add is None:
            res_add = curr
        else:
            res_add = add_str(curr, res_add)

    return res_add


def TEST(s1, s2):
    a1 = int(s1)
    a2 = int(s2)
    res = multiplyTwoStrings(s1, s2)
    print('res', int(res), 'tgt', a1 * a2)


TEST('15', '3')
TEST('13', '13')

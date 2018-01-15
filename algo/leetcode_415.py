""" add strings
"""

def addStrings(num1, num2):
    m = len(num1)
    n = len(num2)
    if m < n: 
        num1, num2 = num2, num1
        m, n = n, m

    digits = []
    zero = ord('0')
    carry = 0
    for i in range(-1, -m-1, -1):
        c1 = ord(num1[i]) - zero
        if i >= -n:
            c2 = ord(num2[i]) - zero
        else:
            c2 = 0
        
        val = c1 + c2 + carry
        carry = int(val >= 10)        
        digits.append(val % 10)
        
    if carry > 0:
        digits.append(carry)
    return ''.join((map(str, digits[::-1])))


def TEST(n1, n2):
    tgt = n1 + n2
    res = int(addStrings(str(n1), str(n2)))
    if res != tgt:
        print('Error', res, 'but expect', tgt)
    else:
        print('Ok')


TEST(121, 1331)

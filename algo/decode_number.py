''' Parse number from string (atoi/f)
'''

def parse_number(s):
    if not s: return 0
    i = 0
    n = len(s)

    # Skip spaces
    while i < n:
        if ' ' != s[i]: break
        i += 1

    # Initial char: '+', '-', '.' or digit
    ch = s[i]
    sign = 1
    float_i = None
    if '+' == ch:
        sign = 1
        i += 1
    elif '-' == ch: 
        sign = -1
        i += 1
    elif '.' == ch: 
        float_i = i
        i += 1
    elif ch < '0' or ch > '9':
        raise ValueError('wrong number format (non-digit)')
    
    # Accumulating digits
    num = 0
    while i < n:
        ch = s[i]
        # Floating point 
        if '.' == ch:
            if float_i is not None:
                raise ValueError('wrong number format (float)')
            float_i = i
        # Trailing spaces, number parsing ended
        elif ' ' == ch:
            for j in range(i, n):
                if ' ' != s[j]:
                    raise ValueError('wrong number format (trailing)')
            break
        # All non-digit chars should be taken care of above
        elif ch < '0' or ch > '9':
            raise ValueError('wrong number format (non-digit)')
        else:
            num = num * 10 + int(ord(ch) - ord('0'))
        i += 1

    # Adjust for floating point
    if float_i:
        shift = i - float_i - 1
        while shift:
            num /= 10
            shift -= 1

    return num * sign


def TEST(s):
    tgt = float(s)
    res = parse_number(s)
    if res != tgt:
        print('Error', res, 'but expect', tgt)
    else:
        print('Ok')


TEST('2.13')
TEST('-2.33')
TEST('1')
TEST('-.32')
TEST(' -.0')
TEST(' +.120  ')

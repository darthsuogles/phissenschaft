''' Validate parentheses
'''

def isValid(s):
    if s is None or 0 == len(s): return True
    stq = []
    match_pair = {
        ']': '[', 
        ')': '(', 
        '}': '{'
    }
    for ch in s:
        try:
            ch_m = match_pair[ch]
            if len(stq) == 0: 
                return False
            if stq[-1] != ch_m:
                return False
            stq = stq[:-1]
        except:
            stq += [ch]
            
    return 0 == len(stq)


def TEST(s, tgt):
    res = isValid(s)
    if res != tgt:
        print("Error", s, ":", res, 'but expect', tgt)
    else:
        print("Ok")


TEST('()', True)
TEST('()()', True)
TEST('()[()', False)
TEST('()[]{}', True)

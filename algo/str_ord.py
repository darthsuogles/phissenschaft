''' Ordering of string 
''' 

def is_str_ord(s_ord, s_txt):
    ''' The first existing ordering must be 
    ''' 
    if not s_txt: return not s_ord

    vocab = set(list(s_ord))
    it = zip(iter(s_ord), (ch for ch in s_txt if ch in vocab))
    if len([p for p in it if p[0] == p[1]]) < len(s_ord):
        return False
    return True

def TEST(s0, s1):
    res = is_str_ord(s0, s1)
    print(res, s0, s1)

TEST('abc', 'abcd')
TEST('abc', 'abdc')
TEST('abc', 'acbdc')

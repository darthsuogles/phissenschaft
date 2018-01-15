''' Find the first occurrance of a pattern in a string
'''

def strstrBruteForce(s, x):
    ''' Compare the substrings with brute force
    '''
    if not s: return -1
    m = len(s)
    n = len(x)
    if n > m: return -1
    for i in range(0, m - n + 1):
        if s[i:(i+n)] == x:
            return i
    return -1


def strstrKMP(s, x):
    ''' Using Knuth-Morris-Pratt
    '''
    patt = x; text = s
    if not patt or not text: 
        return -1

    # First build the pattern lookup table
    tbl = [0] * (1 + len(patt))
    i = 1; j = 0;
    while i < len(patt):
        if patt[i] == patt[j]:
            i += 1; j += 1; tbl[i] = j
        elif 0 == j:
            i += 1
        else:
            j = tbl[j]
    
    # Search over the query text
    inds = []
    i = 0; j = 0;
    while i < len(text):
        if text[i] == patt[j]:
            i += 1; j += 1;
            if len(patt) == j:
                assert(text[i - len(patt): i] == patt)
                return i - len(patt)
        elif 0 == j:
            i += 1
        else:
            j = tbl[j]

    return -1


def TEST(s, x):
    idx_ref = strstrBruteForce(s, x)
    idx_tgt = strstrKMP(s, x)
    assert(idx_ref == idx_tgt)
    print('Ok')
        

print('-------- TEST CASES ----------')
TEST("CodefightsIsAwesome", "IsA")
TEST("CodefightsIsAwesome", "IA")

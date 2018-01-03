
def brute_force(s):
    n = len(s)
    if n <= 1: 
        return False

    for k in range(1, (n // 2) + 1):
        if 0 != (n % k):
            continue        
        curr_sub = s[:k]

        i = k
        while i < n:
            next_sub = s[i: (i + k)]
            if curr_sub != next_sub:
                break
            i += k

        if i < n:
            continue
        return True

    return False


def rep_once(s):
    # Proof: (case: assume non-tiled, but there is a proper matching)
    # Assume there is an overlapping
    # Assume this is the first possible matching position
    # Take the intersection as m
    # We can concat m out of the string
    # until we hit a non-empty substring k smaller than m
    # The string k is a proper prefix & suffix of m
    # And we know that there is one more m in front
    # We can then move the matching left by m - k
    # And still obtain a valid matching
    # This violates the assumption => k must be empty
    # Thus the string is tiled
    return -1 != (s+s)[1:-1].find(s) 


def kmp(patt, text):
    ''' Find all occurrances of @patt in @text
    '''
    if not patt or not text: 
        return []

    # First build the pattern lookup table
    tbl = [0] * (1 + len(patt))
    i = 1; j = 0;
    print(patt)
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
                inds += [(i - len(patt), i)]
                j = tbl[j]
        elif j == 0:
            i += 1
        else:
            j = tbl[j]

    return inds


def gen_test(fn):
    
    def _TEST(s, tgt):
        res = fn(s)
        if tgt != res:
            print("ERROR")
        else:
            print("OK")

    print('testing: {}'.format(fn))
    return _TEST


TEST = gen_test(rep_once)
TEST("abc", False)
TEST("aaa", True)
TEST("abab", True)

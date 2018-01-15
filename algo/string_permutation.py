''' String permutation
'''

def stringPermutation(s):
    ''' Return the list of permutations, in order
    '''

    def find_perms(_chars):
        if not _chars: return []
        if 1 == len(_chars): return [_chars]

        _seen = set()
        res = []  # stores the reversed permutations
        n = len(_chars)  # special care for duplicates
        for i, ch in enumerate(_chars):
            if ch in _seen: continue 
            _seen.add(ch)  # make sure we don't get duplicates
            _chars[0], _chars[i] = _chars[i], _chars[0]
            res += [perm + [ch] for perm in find_perms(_chars[1:])]
            _chars[0], _chars[i] = _chars[i], _chars[0]

        return res

    res = find_perms(sorted(list(s)))
    return [''.join(perm_rev[::-1]) for perm_rev in res]


def TEST(s):
    print('-----------------------')
    res = stringPermutation(s)    
    for perm in res:
        print(perm)


TEST("CBA")
TEST("ABA")

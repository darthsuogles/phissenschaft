''' Matching regular expression with '.' and '*'
'''

def regularExpressionMatching(s, p):
    if not p: return not s  # empty-empty match is okay

    # Check when there is a zero-to-more matching
    if len(p) > 1 and '*' == p[1]:
        if regularExpressionMatching(s, p[2:]):
            return True  # get zero-matching
        p_sub = p
    else:
        p_sub = p[1:]

    if not s: 
        return False

    if s[0] != p[0] and '.' != p[0]:
        return False

    return regularExpressionMatching(s[1:], p_sub)


def regexDP(s, p):
    ''' Matching regular expression
    '''
    m = len(s); n = len(p)
    tbl = []  # stores substring matching with 1-based index
    for i in range(m+1):
        tbl.append([False] * (n+1))

    tbl[0][0] = True  # empty matches empty

    for j in range(1, n + 1):
        if j > 1 and '*' == p[j-1]:
            tbl[0][j] = tbl[0][j-2]  # empty can match zero
        for i in range(1, m + 1):
            a = s[i-1]; b = p[j-1]
            if a == b or '.' == b:
                tbl[i][j] = tbl[i-1][j-1]
                continue
            if '*' == b:
                if '.' == p[j-2] or a == p[j-2]:
                    tbl[i][j] = tbl[i-1][j] or tbl[i-1][j-2]
                tbl[i][j] = tbl[i][j] or tbl[i][j-2]
                continue

            tbl[i][j] = False

    return tbl[m][n]
            

def TEST(s, p, tgt):
    #print('Ok' if regexDP(s, p) == tgt else 'Error')
    print('Ok' if regularExpressionMatching(s, p) == tgt else 'Error')


TEST('bb', 'b', False)
TEST('zab', 'z.*', True)
TEST('a', 'ab*', True)
TEST("ab", '.*..c*', True)
TEST("abdsads", '.*', True)
TEST("abcd", "d*", False)
TEST("aab", "c*a*b*", True)
TEST("", "c*c*", True)
TEST("", "c*c*d", False)

''' Longest palindromic subsequence
'''

def longestPalindromeSubseq(s):
    if not s: return 0
    n = len(s)

    # Check if it is palindrome
    is_palin = True
    for i in range(n // 2):
        if s[i] != s[n-i-1]: 
            is_palin = False; break
    if is_palin:
        return n
        
    # If not, go ahead with DP
    # We are using a more space efficient approach
    tbl = [1] * n; tbl_p1 = [1] * n; tbl_p2 = [1] * n

    for k in range(1, n):
        for i in range(n-k):
            j = i + k   
            val_incl = max(tbl_p1[i], tbl_p1[i+1])
            val_excl = 0
            if i + 1 < j:
                val_excl = tbl_p2[i+1]
            tbl[i] = max(2 * int(s[i] == s[j]) + val_excl, val_incl)

        tbl, tbl_p1, tbl_p2 = tbl_p2, tbl, tbl_p1
    
    print(tbl, tbl_p1, tbl_p2)
    return tbl_p1[0]


def TEST(s):
    res = longestPalindromeSubseq(s)
    print(res)


TEST("bbbab")
TEST("cbbd")
TEST("a")
TEST("aaa")

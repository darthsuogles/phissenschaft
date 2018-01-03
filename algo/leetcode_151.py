''' Reverse words in a string
'''

def reverseWordsOut(s):
    if not s: return ''
    s = s.strip().replace(' ', '$')
    if not s: return ''
    words_rev = ' '.join([w[::-1] for w in s.split('$') if w])
    return words_rev[::-1]


def reverseWords(s):
    if not s: return ''
    s = s.strip()
    if not s: return ''
    n = len(s)
    s = list(s) + [' ']

    # Reverse individual words
    i = 0; j = 0
    while j <= n:
        ch = s[j]
        if ' ' == ch:
            while i < j and ' ' == s[i]: i += 1
            m = j - i
            for p in range(m // 2):
                q = m - p - 1
                s[i + p], s[i + q] = s[i + q], s[i + p]
            i = j
        j += 1
        
    # Remove spaces
    i = 1; j = 1
    while j < n:
        while j < n:
            if ' ' != s[j-1] or ' ' != s[j]:
                break
            j += 1
        if n == j: break
        s[i] = s[j]; i += 1; j += 1
        
    # Removing trailing spaces
    s = s[:i]

    # Reverse the whole string
    n = len(s)
    for i in range(n // 2):
        j = n - i - 1
        s[i], s[j] = s[j], s[i]
    
    return ''.join(s)


def TEST(s):
    print('TEST')
    print('avant: <{}>'.format(s))
    print('apres: <{}>'.format(reverseWords(s)))
    

TEST("  the   sky      is   blue ")

''' Match against a dictionary when at most one char can be missing
'''

def match_with_missing(query, text):
    ''' Can delete any number of characters
    '''
    if not query or not text: return 0

    m = len(query); n = len(text)
    i = 0; j = 0
    while i < m and j < n:
        a = query[i]; b = text[j]
        if a == b: 
            j += 1  # only updated upon matching
        i += 1
        
    return j


def match_approx(query, text, quota=1):
    ''' Query can miss @quota many character(s)
    '''
    if not query or not text: return 0
    m = len(query); n = len(text)
    if m > n + quota or n > m + quota: return 0

    i = 0; j = 0
    while i < m and j < n:
        a = query[i]; b = text[j]
        if a == b: 
            j += 1  # only updated upon matching
        else:
            quota -= 1

        if quota < 0: break
        i += 1
        
    quota -= n - j  # trailing miss-matchings
    if quota < 0: return 0
    return j


print(match_approx('amazon', 'amazons'))
print(match_approx('samazon', 'amazons', 2))
        
        

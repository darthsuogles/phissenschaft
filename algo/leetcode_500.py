''' Words typable in a single keyboard row
'''

kbd = [
    set(list('qwertyuiop')),
    set(list('asdfghjkl')),
    set(list('zxcvbnm')),
]

def findWords(words):
    res = []
    for word in words:
        if not word: continue
        w = word.lower()
        for row in kbd:            
            if w[0] not in row: 
                continue
            if set(w) <= row:
                res += [word]; break
    return res
        


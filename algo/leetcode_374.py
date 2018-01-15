''' Guess number
'''

A = 6
def guess(n):
    if n == A: return 0
    if A < n: return -1
    if n < A: return 1


def guessNumber(n):       
    a = 1; b = n
    while a + 1 < b:
        k = (a + b) // 2
        curr = guess(k)
        if 0 == curr:
            return k
        if 1 == curr:
            a = k
        else:
            b = k

    if 0 == guess(a): 
        return a
    return b


print('guess', guessNumber(10))

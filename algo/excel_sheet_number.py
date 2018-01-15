''' A -> 1; AA -> 27
'''

def excelSheetColumnNumberRec(s):
    if not s: return 0
    
    last = ord(s[-1]) - ord('A') + 1
    prev = excelSheetColumnNumber(s[:-1]) * 26
    return prev + last

def excelSheetColumnNumber(s):
    if not s: return 0

    val = 0
    for ch in s:
        curr = ord(ch) - ord('A') + 1
        val = val * 26 + curr
    
    return val


def excelColumnIndex(num):
    if not num: raise ValueError("num > 0")
    s = []
    while num > 0:
        num -= 1
        v = num % 26        
        s.append(chr(v + ord('A')))
        num //= 26

    return ''.join(s[::-1])

def TEST(s):
    print(excelSheetColumnNumber(s))


TEST('AB')
TEST('ZZ')

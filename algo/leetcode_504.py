""" Base 7 convert
"""

def convertToBase7(num):
    if not num: return "0"
    sign = 1
    if num < 0:
        sign = -1
        num = -num
    
    digits = []
    while num:
        digits.append(num % 7)
        num //= 7

    s = ''.join(map(str, digits[::-1]))
    if -1 == sign:
        s = '-' + s
    return s

print(convertToBase7(100))
print(convertToBase7(-7))

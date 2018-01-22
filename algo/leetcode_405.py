"""
Convert to Hex
"""

def toHex(num):
    if num is None: return ""
    if 0 == num: return "0"
    if num < 0:
        BND = (1 << 32) - 1
        num = (num ^ BND + 1) & BND
    hexes = []
    while num:
        rem = num % 16
        num = num // 16
        if rem < 10:
            hexes.append(chr(ord('0') + rem))
        else:
            hexes.append(chr(ord('a') + rem - 10))
    return ''.join(hexes[::-1])


def TEST(num):
    print(toHex(num))


TEST(26)
TEST(-1)

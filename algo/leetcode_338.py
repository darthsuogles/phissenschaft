''' Counting number of bits
'''

def countBits(num):
    if num < 0: return []

    num_bits = [None] * (num + 1)
    num_bits[0] = 0
    i = 1
    while True:
        j = 0
        while j < i and j + i <= num:
            num_bits[j+i] = 1 + num_bits[j]
            j += 1
        if j + i > num:
            break
        i <<= 1
        
    return num_bits


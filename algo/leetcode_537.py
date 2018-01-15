""" Complex number multiplication
"""

def complexNumberMultiply(a, b):
    if not a: return b
    if not b: return a
    
    def parse_complex(s):
        real, imag = None, None
        sign, num = 1, None
        for ch in s:
            if '0' <= ch and ch <= '9':
                num = num or 0
                num = num * 10 + ord(ch) - ord('0')
                continue            
            
            if num is not None:
                if real is None:
                    real = sign * num
                elif imag is None:
                    imag = sign * num
                sign = 1; num = None

            if ch.isspace(): continue
            if ch == '+': sign = 1
            elif ch == '-': sign = -1
              
        return real, imag

    real_a, imag_a = parse_complex(a)
    real_b, imag_b = parse_complex(b)
    real_res = real_a * real_b - imag_a * imag_b
    imag_res = real_a * imag_b + imag_a * real_b
    return '{}+{}i'.format(real_res, imag_res)


def TEST(a, b):
    print(complexNumberMultiply(a, b))

#print(complexNumberMultiply('1 + -2i', '1 - 3i'))
TEST('1+1i', "1+1i")
TEST('1+-1i', '1+-1i')

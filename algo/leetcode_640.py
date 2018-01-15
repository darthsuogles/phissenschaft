""" Solve simple equation
"""

def solveEquation(equation):
    if not equation: return "No solution"

    coeff = 0
    const = 0
    num = None
    default_sign = 1
    sign = 1
    for ch in equation:
        if '0' <= ch <= '9':
            num = num or 0
            num = num * 10 + int(ord(ch) - ord('0'))
            continue             

        if 'x' == ch:
            coeff += sign * (1 if num is None else num)
            num = None
            sign = default_sign
            continue

        if num is not None:
            const += sign * num
            num = None
            sign = default_sign

        if ch.isspace():
            pass
        elif '+' == ch:
            sign = default_sign
        elif '-' == ch:
            sign = -default_sign
        elif '=' == ch:
            sign = default_sign = -1
        else:
            raise ValueError('do not recognize token: {}'.format(ch))

    if num:
        const += sign * num

    print('coeff', coeff, 'const', const)

    if 0 == coeff:
        if 0 == const:
            return "Infinite solutions"
        else:
            return "No solution"
    else:
        if const % coeff == 0:
            return "x={}".format(-const // coeff)
        else:
            return "x={}".format(-const / coeff)


def TEST(eqn):
    res = solveEquation(eqn)
    print(res)


TEST("x+5-3+x=6+x-2")
TEST("x=x")
TEST("2x=x")
TEST("2x+3x-6x=x+2")
TEST("0x=0")
TEST("x + 2 =  x")

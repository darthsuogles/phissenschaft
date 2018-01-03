''' Different ways to construct parse tree
'''

def diffWaysToCompute(input):
    if input is None or '' == input: return []
    ops = {
        '+': lambda v: v[0] + v[1],
        '-': lambda v: v[0] - v[1],
        '*': lambda v: v[0] * v[1]
    }
    
    res = []
    curr_num = 0
    for i, ch in enumerate(input):
        if ch in ops:
            res_left = diffWaysToCompute(input[:i])
            res_right = diffWaysToCompute(input[(i+1):])
            op = ops[ch]
            for x in res_left:
                for y in res_right:
                    res += [op([x, y])]
            curr_num = 0
        else:
            curr_num = curr_num * 10 + int(ch)

    if [] == res:
        return [curr_num]
    return res


def TEST(input):
    print(diffWaysToCompute(input))


TEST('2-1-1')
TEST('2*3-4*5')

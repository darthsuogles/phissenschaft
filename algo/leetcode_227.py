""" Basic calculator
"""

def calculate(s):
    if not s: return 0
    precedence = {'+': 0, '-': 0, '*': 1, '/': 1}
    num = None
    queue = []; op_stack = []    

    def _should_add_op(op):
        if not op_stack: return True
        if precedence[op] > precedence[op_stack[-1]]:
            return True
        return False

    def _eval():
        assert len(queue) >= 2
        assert op_stack
        op = op_stack.pop()
        b = queue.pop()
        a = queue.pop()
        if '+' == op: res = a + b
        elif '-' == op: res = a - b
        elif '*' == op: res = a * b
        elif '/' == op: res = a / b
        queue.append(res)

    for ch in s:
        if ch.isdigit():
            num = num or 0
            num = num * 10 + ord(ch) - ord('0')
            continue
        else:
            if num is not None:
                queue.append(num)
                num = None

        if ch.isspace():
            continue

        if ch not in precedence:
            raise TypeError('unknown operand {}'.format(ch))

        while True:
            if _should_add_op(ch):
                op_stack.append(ch)
                break
            # Remove op from op stack and compute
            _eval()

    if num is not None:
        queue.append(num)

    while op_stack: _eval()
    
    return queue[0]


def TEST(s):
    res = calculate(s)
    tgt = eval(s)
    if res == tgt:
        print('Ok')
    else:
        print('Error', res, 'but expect', tgt)

TEST('1 + 2 * 3')
TEST('1 + 2 + 3')
TEST('1 + 2 * 3')
TEST('0+0')

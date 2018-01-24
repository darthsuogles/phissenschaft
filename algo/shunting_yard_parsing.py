"""
Shunting yard parsing for arithematic expression

Ref: https://en.wikipedia.org/wiki/Shunting-yard_algorithm
"""

_pred = ['^', ['*', '/'], ['+', '-']]
precedence = {}
for i, ops in enumerate(_pred[::-1]):
    for op in ops:
        precedence[op] = i

def should_add_op(op_tgt, op_ref):
    if op_ref == '(':
        return True
    prec_tgt = precedence[op_tgt]
    prec_ref = precedence[op_ref]
    if prec_tgt > prec_ref:
        return True
    if prec_tgt == prec_ref and '^' == op_tgt:
        return True
    return False


def parse(expr):
    if not expr: return []
    queue = []; op_stack = []
    num = None

    for i, ch in enumerate(expr):
        if ch.isdigit():
            num = num or 0
            num = num * 10 + ord(ch) - ord('0')
            continue
        elif num is not None:
            queue.append(num)
            num = None

        # Valid operator
        if ch in precedence:
            # All the operators with equal or higher precedence
            # shall be evaluated before this
            while op_stack:
                if should_add_op(ch, op_stack[-1]):
                    break
                op = op_stack.pop()
                queue.append(op)

            op_stack.append(ch)

        elif ch == '(':
            op_stack.append(ch)

        # Pop until seeing the first left parenthesis
        elif ch == ')':
            while op_stack:
                op = op_stack.pop()
                if '(' == op:
                    break
                queue.append(op)

    if num is not None:
        queue.append(num)

    return queue + op_stack[::-1]



def eval_rev_polish(rev_polish):
    if not rev_polish: return None
    stack = []
    for tok in rev_polish:
        if tok in precedence:
            a, b = stack[-2:]; stack = stack[:-2]
            tok = '({} {} {})'.format(a, tok, b)
        stack.append(tok)

    return stack[0]


def TEST(expr):
    def canon(s): return eval(s.replace('^', '**'))
    tgt = canon(expr)
    res = canon(eval_rev_polish(parse(expr)))
    if res == tgt:
        print('Ok')
    else:
        print('Error', res, 'but expect', tgt)


TEST('31 + (42 + 1) * 2 ^ 3 ^ 4')
TEST('2 * 2 + 3 * 4')

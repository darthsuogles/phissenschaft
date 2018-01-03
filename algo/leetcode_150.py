''' Reverse Polish
'''

def evalRPN(tokens):
    if not tokens: return 0
    stack = []
    ops = set(['+', '-', '*', '/'])
    for tok in tokens:
        if tok in ops:
            b = stack.pop()
            a = stack.pop()
            res = eval('{:d} {} {:d}'.format(a, tok, b))
            stack.append(int(res))
        else:
            stack.append(int(tok))

    return stack[-1]

expr = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
evalRPN(expr)

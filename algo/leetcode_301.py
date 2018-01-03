''' Remove invalid parentheses
''' 

def rmInvalidOne(s):
    ''' Using a more direct method
    '''
    if not s: return s
    
    stack = []
    for i, a in enumerate(s):
        if '(' == a:
            stack.append(i)
            continue
        if ')' == a:
            if not stack:  # found first extra right
                # Remove any left parentheses, if exists
                return s[:i] + rmInvalidOne(s[(i+1):])
            else:
                stack = stack[:-1]
            continue

    # If there are un-matched left parentheses,
    # find the first one, remove it, and get the rest
    if stack:
        i = stack[0]  # first extra
        return s[:i] + rmInvalidOne(s[(i+1):])

    return s
        

def rmInvalidBFS(s):

    def is_valid(s):
        cnts = 0
        for c in s:
            if c == '(':
                cnts += 1
            elif c == ')':
                cnts -= 1
                if cnts < 0: return False

        return cnts == 0

    level = {s}

    # Performing a breadth first search
    # We know that the first ones will be there
    while True:
        valid_all = list(filter(is_valid, level))
        if valid_all:
            return valid_all
        level = {
            s[:i] + s[i+1:] 
            for s in level 
            for i in range(len(s))
            if s[i] == '(' or s[i] == ')'
        }
            
    

def TEST(s):
    print('----------paren-remv--------')
    #res = removeInvalidParentheses(s)
    res = rmInvalidBFS(s)
    print(s, '->', res, 'bfs')
    res = rmInvalidOne(s)    
    print(s, '->', res, 'direct')


TEST("()())()")
TEST(")(")
TEST("n")
TEST("x(")
TEST(")d))")
TEST("(((k()((")

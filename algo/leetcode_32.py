''' Longest valid parenthesis 
https://leetcode.com/articles/longest-valid-parentheses/
'''

def longestValidParenthesesCounter(s):
    if not s: return 0

    n = len(s)
    max_len = 0
    
    # Forward direction
    cntr = 0
    init = -1
    i = 0
    while i < n:
        if '(' == s[i]:
            cntr += 1
        elif ')' == s[i]:
            cntr -= 1

        if cntr < 0:
            cntr = 0
            init = i
        elif 0 == cntr:
            max_len = max(max_len, i - init)

        i += 1
        
    # Reverse direction
    cntr = 0
    i = n - 1
    fini = n
    while i >= 0:
        if ')' == s[i]:
            cntr += 1
        elif '(' == s[i]:
            cntr -= 1

        if cntr < 0:
            cntr = 0
            fini = i
        elif 0 == cntr:
            max_len = max(max_len, fini - i)

        i -= 1

    return max_len

    

def longestValidParenthesesOneStack(s):
    ''' Using one stack to keep invalid parentheses 
    '''
    if not s: return 0
    n = len(s)
    i = 0
    stack = []

    while i < n:
        ch = s[i]
        if '(' == ch:
            stack.append(i)
        elif ')' == ch:
            if stack and '(' == s[stack[-1]]:
                stack = stack[:-1]
            else:
                stack.append(i)
        i += 1
                
    stack = [-1] + stack + [n]
    max_len = 0
    for p, q in zip(stack[:-1], stack[1:]):
        max_len = max(max_len, q - p - 1)

    return max_len

def longestValidParentheses(s):
    ''' Using two stacks
    '''
    if not s: return 0
    n = len(s)
    i = 0
    stack = []
    intervals = []

    while i < n:
        ch = s[i]
        if '(' == ch:
            stack.append(i)
            i += 1
            continue

        # Skipping excessive right 
        if not stack:
            while i < n:
                if ')' != s[i]:
                    break
                i += 1
            continue

        # Update the merge interval
        j = stack[-1]; stack = stack[:-1]
        while intervals:
            p, q = intervals[-1]
            if q + 1 < j: break
            j = min(j, p)
            intervals = intervals[:-1]

        intervals.append((j, i))
        i += 1

    if not intervals:
        return 0
    
    # Check for excessive '('s
    if stack:
        j = stack[-1]
        p, q = intervals[-1]
        intervals = intervals[:-1]
        if p <= j and j < q:
            p = j + 1
        intervals.append((p, q))

    # Compute maximum length
    max_len = 0
    for p, q in intervals:
        max_len = max(max_len, q - p + 1)
    
    return max_len


def TEST(s, tgt):
    res = longestValidParenthesesCounter(s)    
    if res != tgt:
        print('Error', res, 'but expect', tgt)
    else:
        print('Ok')


TEST('()', 2)
TEST('(()', 2)
TEST(')()())', 4)
TEST(')()(()()))', 8)
TEST('()(()', 2)
TEST('()((())', 4)

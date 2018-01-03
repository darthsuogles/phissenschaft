''' Decode string
'''

def decodeStringRec(s):
    ''' Fully recursive solution
    '''
    if not s: return ''
    res = []
    num_rep = 0

    n = len(s)
    i = 0
    while i < n:
        if '[' == s[i]:
            j = i + 1
            num_parens = 1
            while j < n:
                if s[j] == '[':
                    num_parens += 1
                elif s[j] == ']':
                    num_parens -= 1
                    if 0 == num_parens:
                        break
                j += 1

            res.append(num_rep * decodeStringRec(s[(i+1):j]))
            num_rep = 0
            i = j + 1
            continue
        try:
            d = int(s[i])
            num_rep = num_rep * 10 + d
            i += 1
            continue
        except: pass

        res.append(s[i])
        i += 1

    return ''.join(res)


def decodeString(s):
    if not s: return ''
    stack = []
    res = []
    num_rep = 0
    partial = ''

    for i, ch in enumerate(s):
        if '[' == ch:
            if stack:
                stack[-1][-1] += partial
            stack.append([i, num_rep, ''])
            partial = ''
            num_rep = 0
            continue
        if ']' == ch:
            assert stack
            j, rep, prefix = stack[-1]
            stack = stack[:-1]
            
            curr_s = prefix + partial
            partial = ''
            curr = curr_s * rep
            if stack:
                stack[-1][-1] += curr
            else:
                res.append(curr)
            continue
        try:
            d = int(ch)
            num_rep = num_rep * 10 + d
            continue
        except: pass

        if stack:
            partial += ch
        else:
            res.append(ch)

    return ''.join(res)


def TEST(s):
    print('------------------------')
    print('avant', s)
    print('apres', decodeStringRec(s))
    print('apres', decodeString(s))


TEST('3[a]')
TEST('3[a]2[bc]')
TEST('3[a2[bc]]de')
TEST("sd2[f2[e]g]i")

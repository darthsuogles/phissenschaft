''' Simplify unix file path
'''

def simplifyPath(path):
    if not path: return ''
    stack = []
    curr_seg = ''
    for ch in path:    
        if '/' == ch :
            if '..' == curr_seg:
                stack = stack[:-1]
            elif '.' != curr_seg:
                if curr_seg:
                    stack.append(curr_seg)
            curr_seg = ''            
        else:
            curr_seg += ch

    if curr_seg:
        if '..' == curr_seg:
            stack = stack[:-1]
        else:
            stack.append(curr_seg)

    return '/' + '/'.join(stack)


def TEST(s):
    print(simplifyPath(s))

TEST('/home/a/./x/../b//c/')

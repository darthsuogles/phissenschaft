
def compute_fraction(a, b):
    ''' Compute a fraction representation of a / b    
    '''
    tbl = {}  # existing solutions for loops
    s_repr = []
    
    if a > b:
        s_repr.append(str(a // b))
        a = a % b
    s_repr.append('.')
        
    while a != 0:
        if a in tbl:
            i = tbl[a]
            s_repr = s_repr[:i] + ['('] + s_repr[i:] + [')']
            break

        i = len(s_repr)
        tbl[a] = i
        a *= 10
        r, a = a // b, a % b
        s_repr.append(str(r))

    if s_repr[-1] == '.':
        s_repr.append('0')
    if s_repr[0] == '.':
        s_repr = ['0'] + s_repr
        
    return ''.join(s_repr)



print(compute_fraction(122,7))
print(compute_fraction(7,4))

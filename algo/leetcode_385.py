
def deserialize(s):
    stq = []
    curr = None
    _cnt = 0
    _num = 0
    _sgn = 1
    for ch in s:
        if '[' == ch:
            if curr is not None:
                stq.append(curr)
            curr = []
            continue

        if ',' == ch:
            if 0 != _cnt:
                curr.append(_num * _sgn)
                _num = 0; _cnt = 0; _sgn = 1
            continue

        if ']' == ch:
            if 0 != _cnt:
                curr.append(_num * _sgn)
                _num = 0; _cnt = 0; _sgn = 1
            if [] != stq:
                stq[-1].append(curr)
                curr = stq[-1]
                stq = stq[:-1]
            continue

        if '-' == ch:
            _sgn = -1
            continue
        
        try: 
            a = int(ch)
            _cnt += 1
            _num = _num * 10 + a
        except: 
            print('Warning: unknown character', ch, 'in', s)
            return None

    if _cnt != 0:
        return _num * _sgn
    return curr


def TEST(s):
    print(s)
    print(deserialize(s))
    print('-----------------')


TEST("324")
TEST("[123,[456,[789],7],2]")
TEST("[[]]")
TEST("-3")

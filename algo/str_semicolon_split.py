"""
Split on semi-colon
"""


def split_semicolon(text):
    is_escaped = False
    is_quoted = False
    res = []
    curr_str = ''
    for ch in text:
        # BEGIN: state mutation
        if is_escaped:
            is_escaped = False
        elif ch == '\\':
            is_escaped = True
        elif ch == '"':
            is_quoted = not is_quoted
        # END: state mutation
        elif is_quoted:
            pass  # take everything inside quotes
        elif ch == ';':
            res.append(curr_str)
            curr_str = ""
            continue

        # We don't want to omit any chars
        curr_str += ch

    if curr_str:  # might omit the last blank statement
        res.append(curr_str)

    return res


def split_semicolon_recombine(text):
    if not text:
        return []
    sub_texts = text.split(';')

    def check_balanced(text):
        if not text:
            return True
        is_escaped = False
        is_balanced = True
        for ch in text:
            if is_escaped:
                is_escaped = False
                continue
            if ch == '\\':
                is_escaped = True
            elif ch == '"':
                is_balanced = not is_balanced
        return is_balanced

    cmds_buff = []
    res = []
    for txt in sub_texts:
        if not check_balanced(txt):
            cmds_buff.append(txt)
            if len(cmds_buff) > 1:
                res.append(';'.join(cmds_buff))
                cmds_buff.clear()
        else:
            if cmds_buff:
                cmds_buff.append(txt)
            else:
                res.append(txt)
    assert not cmds_buff
    return res


def TEST(text, num_pieces):
    ref = split_semicolon(text)
    tgt = split_semicolon_recombine(text)
    assert ref == tgt, (ref, '!=', tgt)
    assert ';'.join(ref) == text
    print(text, '\t=>\t', end="[")
    for s in ref:
        print(s, end='][')
    print('EOL]')
    assert len(ref) == num_pieces


print('-------TEST-CASES------')
TEST(r'abc; def', 2)
TEST(r';;;; def', 5)
TEST(r'abc; ";\;"', 2)
TEST(r'abc; ";\\\";\""', 2)
TEST(r'abc; "\\"; ; "\;" and "\;\"\\"', 4)
TEST(r'abc; "\\"; ";" "\;"', 3)

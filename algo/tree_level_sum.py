''' Summing nodes at a given level
'''

def treeLevelSum(tree, k):
    if not tree: return 0
    curr_level = -1
    level_sum = 0
    i = 0
    n = len(tree)
    while i < n:
        ch = tree[i]
        if '(' == ch:
            curr_level += 1
            i += 1
            continue
        if ')' == ch:
            curr_level -= 1
            i += 1
            continue

        val = 0
        j = i
        while j < n:
            ch = tree[j]
            if ch == '(' or ch == ')':
                break
            val = val * 10 + int(ch)
            j += 1

        if curr_level == k:
            level_sum += val

        i = j
    
    
    return level_sum


def TEST(tree, k):
    print(treeLevelSum(tree, k))
    

TEST("(0(5(6()())(14()(9()())))(7(1()())(23()())))", 2)
TEST('(1)', 0)

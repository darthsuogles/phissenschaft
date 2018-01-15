
def countSegments(s):
    if 0 == len(s):
        return 0
    
    cnt = 0
    is_prev_space = True  # a phantom space
    for ch in s:
        if ' ' == ch:
            is_prev_space = True
            continue
        if is_prev_space:
            cnt += 1
            is_prev_space = False

    return cnt

def TEST(s, n):
    res = countSegments(s)
    if res != n:
        print("Error: {} != {}".format(res, n))
    else:
        print("OK")
    
        
    
TEST('   ', 0)
TEST('Hello, my name is John', 5)
TEST("Of all the gin joints in all the towns in all the world,   ", 13)

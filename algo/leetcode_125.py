''' Check palindrome, ignore non-alphanumerics
'''

def isPalindrome(s):
    n = len(s)
    if n <= 1: return True
    i = 0; j = n-1

    while i < j:
        while i < j:
            if s[i].isalnum(): break
            i += 1
        while i < j:
            if s[j].isalnum(): break
            j -= 1
            
        if s[i].lower() != s[j].lower():
            return False
        i += 1; j -= 1

    return True

def TEST(s, tgt):
    res = isPalindrome(s)
    if res != tgt:
        print("Error", res, "but expect", tgt)
    else:
        print("Ok")

TEST("A man, a plan, a canal: Panama", True)
TEST("race a car", False)
TEST("ab", False)

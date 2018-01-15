""" Word pattern
"""

def wordPattern(pattern, words):
    if not words: return not pattern
    if not pattern: return not words
    patt2tok = {}; tok2patt = {}
    words = words.split()
    if len(words) != len(pattern):
        return False
    for tok, patt in zip(words, pattern):
        try:
            if tok != patt2tok[patt]:
                return False
        except:
            if tok in tok2patt:
                return False
            tok2patt[tok] = patt
            patt2tok[patt] = tok

    return True


def TEST(pattern, words, expected): 
    tgt = wordPattern(pattern, words)
    if tgt != expected:
        print('Error', tgt, 'but expect', expected)
    else:
        print('Ok')

print('-------TEST CASES---------')
TEST("abba", "dog cat cat dog", True)
TEST("abba", "dog cat cat fish", False)
TEST("abba", "dog dog dog dog", False)
TEST("aaa", "dog dog dog dog", False)

''' Longest substring sans repeating chars
'''

def lengthOfLongestSubstring(s):
    if not s: return 0
    char_idx = {}
    max_len = 0
    #curr = ''
    for i, a in enumerate(s):
        if a not in char_idx:
            #curr += a
            char_idx[a] = i
            if len(char_idx) > max_len:
                max_len = len(char_idx)
                #print(curr)                
        else:            
            j = char_idx[a]
            char_idx = dict([(ch, k) 
                             for ch, k in char_idx.items() 
                             if k > j])
            char_idx[a] = i
            #print(char_idx)
            curr = s[(j+1):(i+1)]
            
    return max_len


def TEST(s):
    print('----------------')
    print(lengthOfLongestSubstring(s))

TEST('abcabcbb')
TEST('bbbbb')
TEST('pwwkew')
TEST('aab')

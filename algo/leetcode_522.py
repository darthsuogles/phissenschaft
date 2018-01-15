''' Longest word in dictionary with deletion
'''

class TrieNode(object):
    def __init__(self):
        self.is_terminal = False
        self.children = {}

    def search_approx(self, s, depth=0, prefix=''):
        if not s:
            if self.is_terminal: 
                return depth, prefix
            else:
                return 0, ''

        ch = s[0]; s = s[1:]
        # Search next best from current
        curr_max, curr_match = self.search_approx(s, depth, prefix)
        try:
            next = self.children[ch]            
            next_max, next_match = next.search_approx(s, depth + 1, prefix + ch)            
            # Return lexicalgrpahically smaller match
            if next_max > curr_max or (next_max == curr_max and next_match < curr_match):
                curr_max = next_max
                curr_match = next_match            
        except:
            pass
            
        return curr_max, curr_match


    def insert(self, s):
        if not s:
            self.is_terminal = True
            return
        
        ch = s[0]; s = s[1:]
        if ch in self.children:
            next = self.children[ch]
        else:
            next = TrieNode()
            self.children[ch] = next

        next.insert(s)


def findLongestWordTrie(s, d):
    ''' Using a special search method in Trie
    '''
    if not s or not d: return ''
    
    trie = TrieNode()
    for word in d:
        trie.insert(word)

    _, match = trie.search_approx(s)
    return match


def findLongestWord(s, d):
    ''' Use plain old longest common subsequence
    '''
    def lcs(query, text):
        m = len(query); n = len(text)
        i = 0; j = 0        
        while i < m and j < n:
            a = query[i]; b = text[j]
            if a == b: i += 1
            j += 1

        return i

    curr_max = 0
    curr_match = ''
    for word in d:
        if len(word) < curr_max:
            continue
        _m = lcs(word, s)
        # Requires full match for 'word'
        if _m < len(word):
            continue
        if _m > curr_max:
            curr_max = _m
            curr_match = word
        if _m == curr_max and word < curr_match:
            curr_match = word

    return curr_match


def TEST(s, d):
    print(findLongestWord(s, d))


TEST("abpcplea", ["apple","bee","car"])
TEST("aas", ["a", "b", "cc"])
TEST("wordgoodgoodgoodbestword", ["word","good","best","good"])
TEST("aewfafwafjlwajflwajflwafj",
     ["apple","ewaf","awefawfwaf","awef","awefe","ewafeffewafewf"])

""" Word break
"""

class TrieNode(object):
    def __init__(self, is_terminal=False):
        self.is_terminal = is_terminal
        self.children = [None] * 26
        
    def insert(self, s):
        if not s: 
            self.is_terminal = True
            return
        i = ord(s[0]) - ord('a')
        if not self.children[i]:
            node = self.children[i] = TrieNode()
        else:
            node = self.children[i]
        node.insert(s[1:])

    def next(self, ch):
        i = ord(ch) - ord('a')
        return self.children[i]


def wordBreak(s, wordDict):
    if not s: return not wordDict
    trie_root = TrieNode()
    # Reverse search trie
    for word in wordDict:
        trie_root.insert(word[::-1])

    # Dynamic programming
    n = len(s)
    tbl_match = [False] * (n + 1)
    tbl_match[-1] = True
    for i in range(n):
        node = trie_root
        for j in range(i, -2, -1):
            if node.is_terminal and tbl_match[j]:
                tbl_match[i] = True
                break
            node = node.next(s[j])
            if not node: break

    return tbl_match[n-1]


print(wordBreak("leetcode", ["leet", "code"]))
print(wordBreak("leetcode", ["leet", "coder"]))

s = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabaabaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
wordDict = ["aa","aaa","aaaa","aaaaa","aaaaaa","aaaaaaa","aaaaaaaa","aaaaaaaaa","aaaaaaaaaa","ba"]
print(wordBreak(s, wordDict))

"""
Find all concatenated words
"""

class TrieNode(object):
    def __init__(self):
        self.children = [None] * 26
        self.is_terminal = False

    def insert(self, s):
        if not s:
            self.is_terminal = True
            return
        ch = s[0]
        idx = ord(ch) - ord('a')
        node = self.children[idx]
        if node is None:
            node = self.children[idx] = TrieNode()
        node.insert(s[1:])

    def is_concat(self, s, root, prev_seen=0):
        if not s:
            return self.is_terminal and (prev_seen > 0)
        if self.is_terminal:
            if root.is_concat(s, root, prev_seen + 1):
                return True

        idx = ord(s[0]) - ord('a')
        node = self.children[idx]
        if node is None:
            return False
        return node.is_concat(s[1:], root, prev_seen)

    def is_concat_dp(self, s):
        s = s[::-1]
        n = len(s)
        tbl_matched = [False] * (n + 1)
        tbl_matched[-1] = True
        for i in range(n):
            node = self
            for j in range(i, -1, -1):
                idx = ord(s[j]) - ord('a')
                node = node.children[idx]
                if node is None:
                    break
                if node.is_terminal and tbl_matched[j - 1]:
                    tbl_matched[i] = True
                    break

        return tbl_matched[n - 1]

def findAllConcatenatedWordsInADictTrie(words):
    if not words: return []
    root = TrieNode()
    words = sorted(words, key=lambda s: len(s))

    res = []
    for word in words:
        if not word: continue
        if root.is_concat_dp(word):
            res.append(word)
        root.insert(word)

    return res


def findAllConcatenatedWordsInADict(words):
    if not words: return []

    words = sorted(words, key=lambda s: len(s))

    vocab = set()
    def is_concat_dp(s):
        n = len(s)
        tbl_matched = [False] * (n + 1)
        tbl_matched[-1] = True
        for i in range(n):
            for j in range(i, -1, -1):
                if s[j:(i+1)] in vocab and tbl_matched[j - 1]:
                    tbl_matched[i] = True
                    break

        return tbl_matched[n - 1]

    res = []
    for word in words:
        if not word: continue
        if is_concat_dp(word):
            res.append(word)
        vocab.add(word)

    return res


def TEST(words):
    print(findAllConcatenatedWordsInADict(words))


TEST(["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"])

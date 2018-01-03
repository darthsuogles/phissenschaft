"""
Replace words with roots
"""

def replaceWords(root_vocab, sentence):

    class TrieNode(object):
        def __init__(self):
            self.children = [None] * 26
            self.terminal = False

        def insert(self, word):
            if not word:
                self.terminal = True
                return
            ch = word.pop(0)
            idx = ord(ch) - ord('a')
            next_node = self.children[idx]
            if next_node is None:
                next_node = self.children[idx] = TrieNode()
            next_node.insert(word)

        def find(self, word, depth=0):
            if self.terminal:
                return len(word)
            if not word:
                return None
            ch = word.pop(0)
            idx = ord(ch) - ord('a')
            next_node = self.children[idx]
            if next_node is None:
                return None
            return next_node.find(word, depth + 1)

        def pprint(self, depth=0):
            chs = []
            for idx, node in enumerate(self.children):
                if node is None: continue
                chs.append(chr(idx + ord('a')))
                node.pprint(depth + 1)
            _repr = 2 * depth * ' '
            _repr += 'Y' if self.terminal else 'N'
            _repr += '|=> [{}]'.format('|'.join(chs))
            print(_repr)
        
    trie_root = TrieNode()
    for word in root_vocab:
        trie_root.insert(list(word))
    trie_root.pprint()
    
    sentence = sentence.split()
    for i, word in enumerate(sentence):
        suff_len = trie_root.find(list(word))
        if suff_len is not None:
            pref_len = len(word) - suff_len
            sentence[i] = word[:pref_len]

    return ' '.join(sentence)


def TEST(root_vocab, sentence):
    print('----TEST-CASE----')
    res = replaceWords(root_vocab, sentence)
    print(res)

TEST(['cat', 'bat', 'rat'],
     "the cattle was rattled by the battery")

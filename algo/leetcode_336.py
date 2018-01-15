''' Palindromic pairs
'''
            
class Trie(object):

    class TrieNode(object):
        def __init__(self, is_terminal=False, aux=None):
            self.is_terminal = is_terminal
            self.children = {}
            self.aux = aux or []

        def search(self, s, pred, res):
            if not s: 
                if self.is_terminal:
                    res += self.aux
                return

            if self.is_terminal:
                if pred(s):
                    res += self.aux

            ch = s[0] 
            if ch not in self.children:
                return

            self.children[ch].search(s[1:], pred, res)


    def __init__(self, s_coll):
        self.root = self.TrieNode()
        for s, aux in s_coll:
            self.insert(s, aux)


    def insert(self, s, aux):
        self._insert(self.root, s, aux)

    
    def _insert(self, node, s, aux):
        if not s: 
            node.is_terminal = True
            if not node.aux:
                node.aux = [aux]
            else:
                node.aux.append(aux)
            return

        ch = s[0]
        try: 
            next = node.children[ch]
        except: 
            next = self.TrieNode()
            node.children[ch] = next 

        self._insert(next, s[1:], aux)


    def search(self, s, op, res):
        return self.root.search(s, op, res)


def palindromePairs(words):

    def is_palindrome(s):
        if s is None: return True
        n = len(s)
        if 1 == n: return True
        for i in range(n // 2):
            if s[i] != s[n-i-1]: return False
        return True

    root_frw = Trie(
        [(s, i) for i, s in enumerate(words)]
    )
    root_inv = Trie(
        [(s[::-1], i) for i, s in enumerate(words)]
    )
    res = set()

    for i, s in enumerate(words):
        inds = []
        root_frw.search(s[::-1], is_palindrome, inds)
        for j in inds:
            if j == i: continue
            res.add((j, i))

        inds = []
        root_inv.search(s, is_palindrome, inds)
        if not inds: continue
        for j in inds:
            if j == i: continue
            res.add((i, j))

    return list(res)


def palindromePairsHT(words):
    tbl = {}    
    for j, s in enumerate(words):
        tbl[s[::-1]] = j

    def is_palindrome(s):
        n = len(s)
        for i in range(n // 2):
            if s[i] != s[n-i-1]: return False
        return True

    res = set()
    for i, s in enumerate(words):
        for k in range(len(s) + 1):
            s0, s1 = s[:k], s[k:]
            if is_palindrome(s0):                
                try:
                    j = tbl[s1]
                    if j != i: res.add((j, i))
                except: pass                

            if is_palindrome(s1):
                try:
                    j = tbl[s0]
                    if j != i: res.add((i, j))
                except: pass

    return list(res)


def TEST(words, tgt):
    res = sorted([tuple(p) for p in palindromePairs(words)])
    tgt = sorted([tuple(p) for p in tgt])
    if res != tgt:        
        print('Error', res, 'but expect', tgt)
        res = set(res); tgt = set(tgt)
        inds = tgt - res
        if inds:
            print('but missing', inds, [(words[i], words[j]) for (i, j) in inds])
        inds = res - tgt        
        if inds:
            print('excess', inds, [(words[i], words[j]) for (i, j) in inds])
    else:
        print('Ok')


def PERF_TEST(words):
    from timeit import default_timer as timer
    t0 = timer()
    palindromePairs(words)    
    t1 = timer()
    print('elapsed {:.2f} ms'.format((t1 - t0) * 1000))


TEST(["bat", "tab", "cat"], 
     [[0, 1], [1, 0]])
TEST(["abcd", "dcba", "lls", "s", "sssll"], 
     [[0, 1], [1, 0], [3, 2], [2, 4]])
TEST(["a","b","c","ab","ac","aa"], 
     [[0, 5], [1, 3], [2, 4], [5, 0], [3, 0], [4, 0]])
TEST(["a", "aa", "aaa"],
     [[1,0],[0,1],[2,0],[1,2],[2,1],[0,2]])
TEST(["ab","ba","abc","cba"],
     [[0,1],[1,0],[2,1],[2,3],[0,3],[3,2]])
TEST(["bb","bababab","baab","abaabaa","aaba","","bbaa","aba","baa","b"],
     [[0,5],[0,9],[9,0],[5,0],[1,5],[5,1],[2,5],[8,2],[5,2],[4,3],[7,4],[4,8],[6,0],[7,5],[5,7],[8,9],[9,5],[5,9]])

import json
PERF_TEST(json.load(open('leetcode_336.txt', 'r')))

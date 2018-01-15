''' Word ladder
'''

def wordLadder(beginWord, endWord, wordList):
    vocab = set(wordList)
    visited = set()
    queue = [(beginWord, 1)]
    visited.add(beginWord)
    alphabet = [chr(ord('a') + i) for i in range(26)]
    while queue:
        word, n_steps = queue[0]; queue = queue[1:]
        if word == endWord:
            return n_steps
        for i, ch in enumerate(word):
            for a in alphabet:
                if a == ch: continue
                word_next = word[:i] + a + word[(i+1):]
                if word_next in vocab:
                    if word_next in visited:
                        continue
                    queue.append((word_next, n_steps + 1))
                    visited.add(word_next)                    

    return -1


def TEST(beginWord, endWord, wordList):
    print(wordLadder(beginWord, endWord, wordList))


TEST("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"])

"""
Grid search for words
"""

def exist(board, word):
    if not word: return True
    m = len(board)
    if not m: return False
    n = len(board[0])
    if not n: return False

    def search(x, y, pos):
        if board[x][y] != word[pos]:
            return False
        if pos + 1 != len(word):
            return True
        board[x][y] = '$'
        for i in range(max(x-1, 0), min(x+1, m-1)+1):
            for j in range(max(y-1, 0), min(y+1, n-1)+1):
                # No diagonal
                if (i == x) == (j == y):
                    continue
                if search(i, j, pos + 1):
                    return True
        board[x][y] = word[pos]
        return False

    for x in range(m):
        for y in range(n):
            if search(x, y, 0):
                return True
    return False


def TEST(board, word):
    print(exist(board, word))


grid = [
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

TEST(grid, "ABCCED")
TEST(grid, "SEE")
TEST(grid, "ABCB")


TEST([["a"]], "a")
TEST([["a", "b"], ["c", "d"]], "abcd")

""" Count battleships
"""

def countBattleships(board):
    m = len(board)
    if 0 == m: return 0
    n = len(board[0])
    if 0 == n: return 0

    cnts = 0
    for i in range(m):
        for j in range(n):
            if '.' == board[i][j]:
                continue
            if i > 0 and 'X' == board[i-1][j]:
                continue
            if j > 0 and 'X' == board[i][j-1]:
                continue
            cnts += 1

    return cnts

def countBattleshipsErase(board):
    m = len(board)
    if 0 == m: return 0
    n = len(board[0])
    if 0 == n: return 0
    
    def erase(x, y):
        board[x][y] = 'o'
        for i in range(x, m):
            if 'X' != board[i][y]: 
                break
            board[i][y] = 'o'
        for j in range(y, n):
            if 'X' != board[x][j]:
                break
            board[x][j] = 'o'            
                

    cnts = 0
    for i in range(m):
        for j in range(n):
            if 'X' != board[i][j]: 
                continue
            cnts += 1
            erase(i, j)
            
    return cnts


def TEST(board, tgt):
    _board = [list(row) for row in board]
    res = countBattleships(_board)
    print(_board)
    if res != tgt:
        print('ERROR', res, 'but want', tgt)
    else:
        print('Ok')

TEST(['X..X',
      '...X',
      '...X'], 2)

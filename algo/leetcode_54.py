''' Matrix manipulations
'''

def spiralOrderRot(matrix):
    ''' Take the first row and rotate the rest
    '''
    if not matrix: return []
    row = list(matrix.pop(0))
    A_rot = list(zip(*matrix))[::-1]        
    return row + spiralOrder(A_rot)


def spiralOrder(matrix):
    if not matrix: return []
    m = len(matrix)
    n = len(matrix[0])
    if 0 == n: return []

    res = []
    row_lo = 0; row_hi = m - 1;
    col_lo = 0; col_hi = n - 1;
    i = 0; j = 0
    while True:
        for j in range(col_lo, col_hi + 1):
            res += [matrix[row_lo][j]]
        row_lo += 1
        if row_lo > row_hi: break

        for i in range(row_lo, row_hi + 1):
            res += [matrix[i][col_hi]]
        col_hi -= 1
        if col_lo > col_hi: break

        for j in range(col_hi, col_lo - 1, -1):
            res += [matrix[row_hi][j]]
        row_hi -= 1
        if row_lo > row_hi: break

        for i in range(row_hi, row_lo - 1, -1):
            res += [matrix[i][col_lo]]
        col_lo += 1
        if col_lo > col_hi: break

    return res
    

def spiralOrderRec(matrix):
    ''' Recursively tackle the rim, check for single row / column cases
    '''
    if not matrix: return []
    m = len(matrix)
    n = len(matrix[0])
    if 0 == n: return []    

    def spiral_print(A, r0, r1, c0, c1):
        m = r1 - r0 + 1
        n = c1 - c0 + 1
        if 0 == n or 0 == m: return []
        if 1 == m: 
            return A[r0][c0:(c1+1)]  
        if 1 == n: 
            return [row[c0] for row in A[r0:(r1+1)]]

        res = []
        def _p(i, j): 
            res.append(A[i][j])

        i = r0; j = c0;

        while j < c1:
            _p(i, j); j += 1

        while i < r1:
            _p(i, j); i += 1

        while j > c0:
            _p(i, j); j -= 1

        while i > r0:
            _p(i, j); i -= 1

        res += spiral_print(A, 
                            r0 + 1, r1 - 1,
                            c0 + 1, c1 - 1)

        # The final touch
        return res
    
    return spiral_print(matrix, 0, m - 1, 0, n - 1)


def TEST(ms):
    print('-----spiral_print---------')
    A = list(map(list, ms))
    m = len(A)
    n = len(A[0])
    print('matrix'); 
    for r in A: print(r)
    print('spiral')
    print(spiralOrder(A))

    
TEST([
    '0312d',
    'e#*@e',
    'ab22d'])

TEST([
    '0', '1', '2'
])

TEST([
    '012'
])

TEST([
    '01',
    '52',
    '43'
])

TEST([[2,3,4],[5,6,7],[8,9,10],[11,12,13]])

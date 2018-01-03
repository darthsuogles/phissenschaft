''' Longest increasing path
''' 

def longestIncreasingPath(matrix):
    if not matrix: return 0
    m = len(matrix)
    n = len(matrix[0])
    if 0 == n: return 0

    # There will not be any crossing path
    # as that violates the increasing path hypothesis
    tbl = []
    for i in range(m): tbl.append([0] * n)
    
    def search(i, j):
        # if i < 0 or i + 1 > m or j < 0 or j + 1 > n:
        #     return 0
        if tbl[i][j] > 0: return tbl[i][j]
        
        curr = matrix[i][j]
        len_max_excl = 0
        for ii in range(max(i-1, 0), min(i+1, m-1) + 1):
            for jj in range(max(j-1, 0), min(j+1, n-1) + 1):
                if (ii != i) == (jj != j): 
                    continue
                if matrix[ii][jj] > curr:
                    len_max_excl = max(search(ii, jj), len_max_excl)
                                
        len_max = 1 + len_max_excl
        tbl[i][j] = len_max
        return len_max


    curr_max = 0
    for i in range(m):
        for j in range(n):
            curr_max = max(curr_max, search(i, j))

    return curr_max
    


def TEST(nums):
    print(longestIncreasingPath(nums))


TEST([
  [9,9,4],
  [6,6,8],
  [2,1,1]
])


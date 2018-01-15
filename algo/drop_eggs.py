# This program computes the minumim of worst case number of drops needed 
# to solve to egg problem. Egg problem: given a N storeyed building, find 
# the level from which the eggs break if dropped. 

# S(k, n) :- number of drops needed when k eggs are left while there are n storeys atop

# The recursive algorithm
from collections import defaultdict

def eggs(k, n, _rec_tbl={}):    
    if n == 0 or n == 1:
        # If there is only one more floor,
        # it is definitely the breaking floor
        return 0
    if k == 1:
        # If we only have one egg, 
        # there might or might not be a breaking point
        # thus it takes a total of n tries
        return n

    try: return _rec_tbl[(k, n)]
    except: pass

    # There are a set of floors to choose which floor to 
    # start with. For each floor there is a possibility
    # that it is the breaking floor
    curr = max(1, 1 + eggs(k, n - 1))
    for m in range(2, n + 1):
        curr = min(curr, 
                   1 + max(eggs(k - 1, m - 1, 
                                _rec_tbl),  # moving down
                           eggs(k, n - m,
                                _rec_tbl)   # moving up
                   ))
    
    _rec_tbl[(k, n)] = curr
    return curr


# Use dynamics programming to solve the problem more efficiently
def dp_eggs(K, N):

    # Initialize the first row
    S = [range(N)]

    for k in range(1, K):
        curr_level = [0] * N
        S += [curr_level]
        for n in range(1, N):
            curr = max(1, 1 + S[k][n-1])            
            for m in range(1, n):
                curr = min(curr, 1 + max(S[k-1][m-1], S[k][n-m]))        
            S[k][n] = curr

    return S[K-1][N-1]

if __name__ == "__main__":

    print 'Dropping eggs in a 100 storeyed building'
    print 'Number of drops needed with 2 eggs:', dp_eggs(2, 100)
    print 'Number of drops needed with 3 eggs:', dp_eggs(3, 100)

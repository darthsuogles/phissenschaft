'''
'''

def maxset(A):
    if not A: return []

    n = len(A)
    i = 0
    while i < n:
        if A[i] >= 0: 
            break
        i += 1

    if n == i: return []
    max_sum = -1
    max_subarr = []
    curr = []
    curr_sum = 0
    while i < n:
        a = A[i]; i += 1
        if a < 0:
            if curr_sum > max_sum:
                max_sum = curr_sum
                max_subarr = curr
            curr_sum = 0
            curr = []
        else:
            curr_sum += a
            curr += [a]
            
    if curr_sum > max_sum:
        return curr

    return max_subarr



def TEST(arr):
    print(maxset(arr))


TEST([1,2,5,-7,2,3])
TEST([1,2,5,-7,2,5])
TEST([0,0,-1,0])
TEST([336465782, -278722862, -2145174067, 
      1101513929, 1315634022, -1369133069, 
      1059961393, 628175011, -1131176229, -859484421])

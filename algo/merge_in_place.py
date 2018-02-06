''' Merge sorted lists in place
''' 

def mergeLists(A, B):
    n = len(B)
    assert len(A) == 2 * n
    # Merging from last
    i = n - 1; j = n - 1
    k = 2 * n - 1
    while i >= 0 and j >= 0:
        if A[i] > B[j]:
            A[k] = A[i]
            k -= 1
            i -= 1
        else:
            A[k] = B[j]
            k -= 1
            j -= 1
            
    while i >= 0:
        A[k] = A[i]
        k -= 1
        i -= 1
    
    while j >= 0:
        A[k] = B[j]
        k -= 1
        j -= 1

A = [1,3,7,10,15,None,None,None,None,None]
B = [1,3,4,5,6]
mergeLists(A, B)

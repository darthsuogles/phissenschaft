""" Arithmetic Slices
""" 

def numberOfArithmeticSlices(A):
    n = len(A)
    if n < 3: return 0    

    counts = 0
    i = 0
    while i + 2 < n:
        j = i + 1
        while j + 1 < n:
            if A[j] - A[j-1] != A[j+1] - A[j]:
                break
            j += 1
        
        m = j - i - 1
        if 0 != m:
            counts += (m + 1) * m // 2
        i = j

    return counts


print(numberOfArithmeticSlices([1,2,3,4,5,7,7,7]))

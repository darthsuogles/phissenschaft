''' Separating positive and negative elements O(1)
'''

def sep_pos_neg(arr):
    if not arr: return None
    n = len(arr)
    for i in range(n-1, -1, -1):
        if arr[i] >= 0: continue
        j = i + 1
        for j in range(i+1, n):
            if arr[j] < 0: break
            tmp = arr[j-1]
            arr[j-1] = arr[j]
            arr[j] = tmp
            
    return arr



        
    
res = sep_pos_neg([1,-2,3,-4,5])
print(res)

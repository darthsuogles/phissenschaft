''' Find union and intersection of two sorted arrays
'''

def find_union_intersec(arr1, arr2):
    if not arr1: return arr2, arr2
    if not arr2: return arr1, arr1

    union_buf = []
    intsec_buf = []
    i = 0; j = 0
    m = len(arr1); n = len(arr2)
    while i < m and j < n:
        a = arr1[i]; b = arr2[j]
        if a <= b:
            union_buf += [a]
            i += 1
            if a == b:
                intsec_buf += [a]
                j += 1; 

        else:
            union_buf += [b]
            j += 1
            continue

    while i < m:
        union_buf += [arr1[i]]
        i += 1
    
    while j < n:
        union_buf += [arr2[j]]
        j += 1

    return union_buf, intsec_buf
        


def TEST(arr1, arr2):
    print('------------------------')
    ubf, ibf = find_union_intersec(arr1, arr2)
    print('     union', ubf)
    print('intsection', ibf)


TEST([1,3,4,5,7], [2,3,5,6])

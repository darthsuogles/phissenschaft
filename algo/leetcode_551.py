""" Check attendance record
""" 

def checkRecord(s):
    if not s: return True
    
    st_abs = 0
    st_late = 0
    for ch in s:
        if 'L' == ch:
            if st_late >= 2: return False
            st_late += 1
            continue

        st_late = 0
        
        if 'A' == ch:
            if st_abs >= 1: return False
            st_abs += 1
            continue

    return True


print(checkRecord("PPALLP"))
print(checkRecord("PPALLL"))

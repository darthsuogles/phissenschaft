''' Remove k digits to create smallest number
'''

def get_smallest(num, k, tbl={}):
    if 0 == k or 0 == num: return num
    try: return tbl[(num, k)]
    except: pass
    val_sans = get_smallest(num // 10, k - 1, tbl)
    val_with = get_smallest(num // 10, k, tbl) * 10 + num % 10    
    val = min(val_sans, val_with)
    tbl[(num, k)] = val
    return val

print(get_smallest(1432219, 3))

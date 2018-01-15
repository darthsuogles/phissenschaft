''' Compare verion numbers
'''

def compareVersion(version1, version2):
    vs1 = version1.split('.')
    vs2 = version2.split('.')
    n = len(vs1)
    m = len(vs2)
    i = 0; j = 0
    while i < n and j < m:
        a = int(vs1[i]); b = int(vs2[j])
        if a > b: return 1
        if a < b: return -1
        i += 1; j += 1
        
    while i < n:
        if int(vs1[i]) != 0:
            return 1
        i += 1

    while j < m:
        if int(vs2[j]) != 0:
            return -1
        j += 1

    return 0


def TEST(vs1, vs2, tgt):
    res = compareVersion(vs1, vs2)
    if res == tgt:
        print('Ok')
    else:
        print('Error', res, 'but expect', tgt)

TEST('0', '1', -1)
TEST('0', '0.1', -1)
TEST('1.0', '1', 0)
TEST('01', '1', 0)

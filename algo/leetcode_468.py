'''  Validate IP Address
'''

def validIPAddress(IP):
    ver = None
    if IP is None or 0 == len(IP): 
        return 'Neither'
    valid_chars = set('0123456789abcdefABCEDF')
    
    def check_v4(seg):
        #print('IPv4 seg', seg)
        if seg is None or 0 == len(seg): 
            return False
        if len(seg) > 3: return False
        if '0' == seg[0]: return 1 == len(seg)
        num = 0
        for ch in seg:
            if ch < '0' or ch > '9': 
                return False
            num = num * 10 + int(ch)
            if num > 255: return False
        return True

    def check_v6(seg):
        #print('IPv6 seg', seg)
        if seg is None or 0 == len(seg): return False
        if len(seg) > 4: return False
        for ch in seg:
            if ch not in valid_chars:
                return False
        return True

    check_seg = None
    num_segs = 0
    curr_seg = ''
    for a in IP:
        if a in valid_chars:
            curr_seg += a
            continue

        elif '.' == a:
            if ver is None:
                ver = 4
                check_seg = check_v4
            elif 4 != ver: 
                return 'Neither'

        elif ':' == a:
            if ver is None:
                ver = 6
                check_seg = check_v6
            elif 6 != ver:
                return 'Neither'

        else:
            return 'Neither'
        
        if not check_seg(curr_seg):
            return 'Neither'
        num_segs += 1
        curr_seg = ''

    if '' == curr_seg: 
        return 'Neither'
    num_segs += 1
    if 4 == ver and num_segs != 4: 
        return 'Neither'
    if 6 == ver and num_segs != 8:
        return 'Neither'

    if not check_seg(curr_seg):
        return 'Neither'
        
    return 'IPv{}'.format(ver)
    
            
def TEST(IP, tgt_ver):
    tgt = 'Neither' if -1 == tgt_ver else 'IPv{}'.format(tgt_ver)
    res = validIPAddress(IP)
    if tgt != res:
        print("ERROR", IP, res, 'should be', tgt)
    else:
        print("OK")


TEST('172.16.254.1', 4)
TEST('172.16.254.1.', -1)
TEST('0172.16.254.1.', -1)
TEST('0172.16:254.1.', -1)
TEST('256.256.256.256', -1)
TEST('2001:0db8:85a3:0:0:8A2E:0370:7334', 6)
TEST('2001:00db8:85a3:0:0:8A2E:0370:7334', -1)
TEST('02001:0db8:85a3:0000:0000:8a2e:0370:7334', -1)
TEST("12.12.12.12.12", -1)
TEST("192.0.0.1", 4)

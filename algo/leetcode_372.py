''' a ^ b == ? (mod 1337)
'''

n = 1336
bnd = int(n ** 0.5) + 2

is_prime_tbl = [True] * bnd
is_prime_tbl[0] = False
is_prime_tbl[1] = False
is_prime_tbl[2] = True
for a in range(2, bnd):
    if not is_prime_tbl[a]:
        continue
    s = a + a
    while s < bnd:
        is_prime_tbl[s] = False
        s += a
    
prime_tbl = [a for a, is_prime in enumerate(is_prime_tbl) if is_prime]

# Remainder of 1337, 1337 = 7 * 191
def superPow(a, b):
    def find_mod_kern(intp, j, m):
        if [] == intp: return 0
        if 0 == j: return intp[0] % m
        r = find_mod_kern(intp, j - 1, m) 
        return (r * 10 + intp[j]) % m

    def find_mod(intp, m):
        if [] == intp: return 0
        return find_mod_kern(intp, len(intp) - 1, m)

    pow_tbl = {}
    def sPow(a, h, n):
        if 0 == h: return 1
        try: return pow_tbl[h]
        except: pass
        p = sPow(a, h // 2, n)
        res = (p * p) % n
        if 1 == h % 2:
            res = (res * a) % n
        pow_tbl[h] = res
        return res


    return sPow(a, find_mod(b, 1140), 1337)

print(superPow(2, [3,3,3,3,3,3,3,1,9,3], 1337))

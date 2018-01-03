""" Partition palindrome
"""

def partition(s):
    n = len(s)
    if n < 2: return [[s]]
    from collections import defaultdict
    palin_tbl = defaultdict(list)
    for init in range(0, n - 1):
        for fini in range(init + 1, n):
            i = init; j = fini
            while i < j:
                if s[i] != s[j]:
                    break
                i += 1; j -= 1
            if i >= j:
                palin_tbl[init].append(fini)
    
    def search(i, mem):
        if i >= n: return [[]]
        try: return mem[i]
        except: pass
        _palin_i = [i] + palin_tbl[i]
        _combs = []
        for j in _palin_i:
            _combs_next = search(j + 1, mem)            
            _curr = s[i:(j+1)]
            for comb in _combs_next:
                _combs.append([_curr] + comb)
        mem[i] = _combs
        return _combs
        
    mem = {}
    return search(0, mem)

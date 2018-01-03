"""
Create exact number of 'A's with copy-and-paste
"""

def minSteps(n):
    if 1 == n: return 0

    # Get all the prime numbers below n
    prime_tbl = [True] * n
    for p in range(2, n):
        if not prime_tbl[p]:
            continue
        for q in range(p*p, n, p):
            prime_tbl[q] = False
    
    # Get prime factors
    factors = [p for p in range(2, n) if prime_tbl[p]]
    
    steps_tbl = [None] * (n + 1)
    
    def search(k):
        _r = steps_tbl[k]
        if _r is not None:
            return _r
        min_steps = k
        for p in factors:
            if p * p > k: break
            rem = k // p
            if rem * p != k: continue
            # 1 copy + (p - 1) pastes
            curr_steps = p + search(rem)
            min_steps = min(curr_steps, min_steps)

        steps_tbl[k] = min_steps
        return min_steps
    
    return search(n)

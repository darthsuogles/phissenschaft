"""
Get the kth smallest inverse of prime factors
"""

def get_kth_smallest(k: int, primes: list):
    if not primes: return -1
    inds = [1] * len(primes)
    def lt(i, p, j, q): return i * q < j * p
    def eq(i, p, j, q): return i * q == j * p

    cnts = 0
    while cnts < k:
        min_denom, min_numer = inds[0], primes[0]
        for denom, numer in zip(inds, primes):
            if lt(denom, numer, min_denom, min_numer):
                min_denom, min_numer = denom, numer

        for i, numer in enumerate(primes):
            denom = inds[i]
            if eq(denom, numer, min_denom, min_numer):
                inds[i] += 1
                cnts += 1

    return min_denom, min_numer


get_kth_smallest(2, [2, 3, 5])

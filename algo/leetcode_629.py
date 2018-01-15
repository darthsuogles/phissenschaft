""" K inverse pairs array
"""

def kInversePairsRef(N, K):
    MOD = 10**9 + 7
    ds = [0] + [1] * (K + 1)
    for n in range(2, N+1):
        novo = [0]
        for k in range(K+1):
            diff = ds[k+1]
            if k >= n:
                diff -= ds[k-n+1]
            novo.append( (novo[-1] + diff) % MOD )
        ds = novo
    return (ds[K+1] - ds[K]) % MOD


def kInversePairsCDF(N, K):
    """ Building up a cumulative distribution
    """
    MOD = 10**9 + 7
    cdf_curr = [0] + [1] * (K + 1)
    cdf_next = [0] + [None] * (K + 1)
    for n in range(2, N+1):
        for k in range(K+1):
            diff = cdf_curr[k+1]
            if k >= n:
                diff -= cdf_curr[k-n+1]
            cdf_next[k+1] = (cdf_next[k] + diff) % MOD
        cdf_next, cdf_curr = cdf_curr, cdf_next
    return (cdf_curr[K+1] - cdf_curr[K]) % MOD


def kInversePairs(N, K):
    """ Building up with pdf
    """
    MOD = 10**9 + 7
    pdf_curr = [1] + [0] * K
    pdf_next = [None] * (K + 1)
    for n in range(1, N+1):
        psum = 0
        for k in range(K+1):
            psum += pdf_curr[k]
            if k >= n:
                psum -= pdf_curr[k-n]
            pdf_next[k] = psum % MOD
        print(pdf_curr)
        pdf_next, pdf_curr = pdf_curr, pdf_next
    return pdf_curr[K] % MOD


def TEST(n, k):
    tgt, ref = kInversePairs(n, k), kInversePairsRef(n, k)
    print('n {}, k {}: num pairs'.format(n, k), tgt, ref)
    assert(ref == tgt)

print('---TEST-CASES---')
TEST(3, 1)
TEST(3, 2)
TEST(4, 2)

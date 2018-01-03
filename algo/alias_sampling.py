''' Alias sampling
'''

from random import random, randint
from collections import defaultdict
from bisect import bisect_left

def gen_rand_arr(k: int):
    if 1 == k: return [1.0]
    p = random()
    return [p * q for q in gen_rand_arr(k - 1)] + [1 - p]

k = 10
#ps = sorted(enumerate(gen_rand_arr(k)), key=lambda x: x[1])
ps = list(randint(10, 30) for i in range(k)); _sps = sum(ps)
ps = sorted(enumerate([a / _sps for a in ps]), key=lambda x: x[1])

class AliasSampler(object):
    
    def __init__(self, discrete_probs):        
        assert discrete_probs, 'must provide valid probability vector'        
        K = len(discrete_probs)
        bar = 1.0 / K
        qs = []
        ps = discrete_probs
        while ps:
            i, p = ps[0]
            ps = ps[1:]
            if bar == p or not ps:
                qs.append([(i, bar)])
                continue

            j, q = ps[-1]
            qs.append([(i, p), (j, bar - p)])
            ps = ps[:-1]
            q -= bar - p
            if not ps:
                qs.append([(j, q)])
            else:
                t = bisect_left([v for i, v in ps], q)    
                ps = ps[:t] + [(j, q)] + ps[t:]

        self.K = K
        self._qs = qs        
        assert abs(1. - sum(p for v_sub in qs for i, p in v_sub)) < 1e-7, \
            'probability must sum up to original'

        ps = defaultdict(lambda: 0)
        for v_sub in qs:
            for i, p in v_sub:
                ps[i] += p
        assert dict(discrete_probs) == ps, \
            'converted probability must be equal to original'


    def sample(self):
        t = randint(0, self.K - 1)
        curr = self._qs[t]
        if len(curr) == 1:
            return curr[0][0]
        (i, p), (j, q) = curr
        _r = random() * (p + q)
        return i if _r < p else j
    
    

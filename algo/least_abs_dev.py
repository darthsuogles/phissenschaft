''' Least absolute deviance for l1 (lp) minimization
   
    \argmin_{\beta} \| x \beta - y \|_1
'''

import numpy as np
from scipy import optimize as optim
import matplotlib.pyplot as plt

def least_abs_regr_iter(x, y):
    ''' Solving simple 1-D problem |x\beta - y| iteratively
    '''
    import numpy as np
    n = x.shape[0]
    max_iters = 500
    beta = 0  # initial guess

    for it in range(max_iters):
        a = 0.0; b = 0.0
        for i in range(n):
            x_i = x[i]; y_i = y[i]
            val = abs(x_i * beta - y_i)
            if 0.0 == val: 
                continue
            a += x_i * x_i / val
            b += x_i * y_i / val

        assert a != 0.0, 'divide-by-zero detected'
        beta = b / a        
        r = np.sum(np.abs(x * beta - y))
        if 0 == it % 100:
            print('iter {:04d}, beta {:.4f}, residual {:.6f}'
                  .format(it, beta, r))
        
    return beta


def irls(x, y, max_iters=3000, delta=1e-4, diff_thresh=1e-7):
    ''' Iteratively re-weighted least square
    '''
    import numpy as np

    n = x.shape[0]
    delta_n = delta * np.ones(n)  # avoid divide-by-zero

    beta = 0.0
    beta_last = beta
    w = np.ones(n)  # initialize weights to one
    r_last = np.sum(np.abs(x * beta - y))
    for i in range(max_iters):        
        ###
        A = np.sum(x * w * x)
        beta = (1.0 / A) * np.sum(x * w * y)
        w = 1.0 / np.maximum(delta_n, np.abs(y - x * beta))
        ### 
        r = np.sum(np.abs(x * beta - y))
        if 0 == i % 100:
            print(i, 'beta', beta, 'r', r)
        if abs(r - r_last) < diff_thresh: 
            break
        if abs(beta - beta_last) < diff_thresh:
            break
        r_last = r
        beta_last = beta

    return beta

def gen_test_samples(n: int):
    x = np.random.randn(n)
    y = x * 0.9 + np.random.randn(n) * 2.7
    neg_inds = x < 0
    x = np.abs(x)
    y[neg_inds] = -y[neg_inds]
    def obj_fn(beta): return np.sum(np.abs(x * beta - y))
    return x, y, obj_fn

x, y, obj_fn = gen_test_samples(7)

betas = y[x != 0] / x[x != 0]
i_opt = np.argmin(list(map(obj_fn, betas)))
beta_opt = betas[i_opt]

beta_opt = optim.minimize(obj_fn, 0).x[0]

plt.plot([0, x.max() + 1], [0, (x.max() + 1) * beta_opt])
plt.plot(x, y, 'o')
plt.show()

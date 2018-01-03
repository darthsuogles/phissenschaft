""" Stock with cooldown
""" 

def maxProfit(prices):
    """ Each position contains a latent state
        {buy, sell, none}
    """
    if not prices: return 0    
    s_none = 0  # no buy-sell
    s_buy = -prices[0]  # bought
    s_sell = -prices[0] - 1  # sold (but no)
    for p in prices[1:]:
        c0 = max(s_none, s_sell)
        c1 = max(s_buy, s_none - p)
        c2 = s_buy + p
        s_none = c0; s_buy = c1; s_sell = c2
    return max(s_none, max(s_buy, s_sell))

def maxProfitSQ(prices):
    if not prices: return 0
    n = len(prices)
    tbl = [None] * n

    def _profit(idx):
        if idx >= n: return 0
        if tbl[idx] is not None: 
            return tbl[idx]
        while idx + 1 < n:
            if prices[idx] < prices[idx + 1]:
                break
            idx += 1
        p_min = prices[idx]
        max_prof = 0
        for i, p in enumerate(prices[idx:], idx):
            p_min = min(p_min, p)
            curr = p - p_min + _profit(i+2)
            max_prof = max(max_prof, curr)

        tbl[idx] = max_prof
        return max_prof
    
    return _profit(0)


def TEST(prices, tgt):
    res = maxProfit(prices)
    if tgt != res:
        print('Error', res, 'but expect', tgt)
    else:
        print('Ok')

TEST([1, 2, 3, 0, 2], 3)
TEST([2, 1], 0)
TEST([2, 4, 1, 7], 6)
TEST([6,1,6,4,3,0,2], 7)

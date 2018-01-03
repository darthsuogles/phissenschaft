"""
Coin change
"""

def coinChange(coins, amount):
    """ Breadth first search
    """
    rec_tbl = [None] * (amount + 1)
    steps = 0
    curr_set = set([0])
    next_set = set()
    while curr_set and steps <= amount:
        for a in curr_set:
            if rec_tbl[a] is not None:
                continue
            rec_tbl[a] = steps
            for c in coins:
                nxt = a + c
                if nxt == amount:
                    return steps + 1
                if nxt < amount:
                    next_set.add(nxt)

        steps += 1
        curr_set = next_set - curr_set
        next_set.clear()

    res = rec_tbl[amount]
    return res if res is not None else -1


def coinChangeRef(coins, amount):
    """ standard dp """
    MAX_VAL = amount + 1
    rec_tbl = {}

    def search(nc, tot_orig):
        if 0 == nc:
            return 0 if 0 == tot_orig else -1
        try: return rec_tbl[(nc, tot_orig)]
        except: pass
        coin = coins[nc - 1]
        min_exch = MAX_VAL
        coin_cnts = 0
        tot = tot_orig
        while tot > 0:
            curr = search(nc - 1, tot)
            if curr != -1:
                min_exch = min(min_exch, curr + coin_cnts)
            coin_cnts += 1
            tot -= coin
        if 0 == tot:
            min_exch = min(min_exch, coin_cnts)

        if min_exch == MAX_VAL:
            min_exch = -1
        rec_tbl[(nc, tot_orig)] = min_exch
        return min_exch

    return search(len(coins), amount)


def TEST(coins, amount):
    ref = coinChangeRef(coins, amount)
    tgt = coinChange(coins, amount)
    assert ref == tgt, ('ref', ref, '!=', tgt)
    print('Ok')


print('----TEST-CASES------')
TEST([1,2,5], 11)
TEST([2], 3)
#TEST([3,7,405, 436], 8839)

print(coinChange([416,157,454,343], 1761))

''' Coin change
'''

def change(amount, coins):
    if 0 == amount: return 1
    if not coins: return 0

    tbl_changes = []
    for a in coins:
        tbl_changes += [[None] * (amount + 1)]

    def find_changes(amount, coins):
        if 0 == amount: return 1
        if not coins: return 0

        n = len(coins)
        record = tbl_changes[n - 1][amount]
        if record is not None:
            return record
        c = coins[-1]
        coins = coins[:-1]
        num_combs = 0
        W = amount
        while W > 0:        
            num_combs += find_changes(W, coins)
            W -= c
        if 0 == W:
            num_combs += 1
            
        tbl_changes[n - 1][amount] = num_combs
        return num_combs

    return find_changes(amount, sorted(coins))


def change(amount, coins):
    if 0 == amount: return 1
    if not coins: return 0

    n = len(coins)
    tbl_changes = [0] * (amount + 1)
    tbl_changes[0] = 1
    for i, c in enumerate(coins, 1):
        for a in range(c, amount + 1):
            tbl_changes[a] += tbl_changes[a - c]
    
    return tbl_changes[amount]


def TEST(amount, coins):
    print(change(amount, coins))

TEST(3, [2])
TEST(5, [1,2,5])
TEST(0, [])
TEST(5000, [11,
            24,
            37,
            50,
            63,
            76,
            89,
            102])

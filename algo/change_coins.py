''' Change coins
'''

def ways_to_change(num, coins):
    if 0 == num:
        # We found all ways to change coins
        # Thus no need for additional coins
        return [[0] * len(coins)]
    if not coins:
        # No coin types left, thus no way to make a change
        return []

    ways = []
    coin_val = coins[-1]
    coins = coins[:-1]
    coin_cnt = 0
    while num >= 0:
        sub_ways = ways_to_change(num, coins)
        ways += [one_way + [coin_cnt] for one_way in sub_ways]
        num -= coin_val
        coin_cnt += 1

    return ways


print(ways_to_change(11, [2,5,10]))

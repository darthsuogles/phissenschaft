''' Best time to buy and sell stock (only once)
'''

def maxProfit(prices):
    n = len(prices)
    if n <= 1: return 0  # cannot do two transactions
    # Need largest of current value
    min_price = prices[0]
    max_profit = 0
    for curr_price in prices[1:]:
        max_profit = max(curr_price - min_price, max_profit)
        min_price = min(min_price, curr_price)

    return max_profit


def TEST(prices):
    print(maxProfit(prices))

TEST([7, 1, 5, 3, 6, 4])

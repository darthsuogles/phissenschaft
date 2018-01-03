''' Best time to buy and sell stocks (unlimited transactions)
'''

def maxProfit(prices):
    ''' This assumes that you can buy-sell "on the same day"
    '''
    if not prices: return 0
    n = len(prices)
    if n < 2: return 0
    
    profit = 0
    for i in range(n - 1):
        diff = prices[i+1] - prices[i]
        if diff > 0:
            profit += diff

    return profit

def maxProfitMinTrans(prices):
    ''' This actually gives a solution with the minimum 
        number of transactions need to achieve max profit
    '''
    if not prices: return 0
    n = len(prices)
    if n < 2: return 0
    i = 0    
    
    # Find the first local minimum
    while i + 1 < n:
        if prices[i] < prices[i+1]:
            #print('BUY', prices[i])
            break
        i += 1
    hold = prices[i]; i += 1
    
    profit = 0
    while i + 1 < n:
        # Find local maximum for sell
        while i + 1 < n:
            a, b, c = prices[i-1], prices[i], prices[i+1]
            if a <= b and b >= c and not (a == b and b == c):
                profit += prices[i] - hold
                #print('SELL', prices[i])
                hold = None
                break
            i += 1
        
        # Find local minimum to buy
        while i + 1 < n:
            a, b, c = prices[i-1], prices[i], prices[i+1]        
            if a >= b and b <= c and not (a == b and b == c):
                hold = prices[i]
                #print('BUY', hold)
                break
            i += 1
            
    if hold is not None and i < n:
        profit += prices[i] - hold
    
    return profit


def TEST(prices):
    print('max profit multiple transactions:', 
          maxProfit(prices))

TEST([7, 1, 5, 3, 6, 4])


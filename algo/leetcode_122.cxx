#include <iostream>
#include <vector>

using namespace std;

int maxProfit(vector<int>& prices) {
    if (prices.size() < 2) return 0;

    int profit = 0;    
    int curr_hold = -1;
    int idx = 0;
    while (true) {
        for (; idx + 1 < prices.size(); ++idx) {
            if (prices[idx] < prices[idx+1]) {
                curr_hold = prices[idx];
                cout << "BUY " << prices[idx] << endl;
                ++idx; break;
            }
        }
        for (; idx + 1 < prices.size(); ++idx) {
            if (prices[idx] > prices[idx + 1]) {
                profit += prices[idx] - curr_hold;
                cout << "SEL " << prices[idx] << endl;
                curr_hold = -1;
                ++idx; break;
            }
        }
        if (idx + 1 == prices.size()) {
            if (-1 != curr_hold) {
                cout << "SEL " << prices[idx] << endl;
                profit += prices[idx] - curr_hold;
            }
            break;
        }
    }    
    return profit;
}

void TEST(vector<int> prices, int tgt) {    
    int res = maxProfit(prices);
    if (res != tgt)
        cout << "RES " << res << " != TGT " << tgt << endl; 
    else 
        cout << "OK" << endl;
}

int main() {
    TEST({8,4,1,2,3,3,2,1,5,6,7,2,3,1,2,2}, 10);
    TEST({2,1,2,0,1}, 2);
    TEST({2,1}, 0);
    TEST({1,2}, 1);
}

/**
 * https://www.hackerrank.com/challenges/equal/problem
 */
#include <iostream>
#include <vector>
#include <unordered_set>

using namespace std;

#define INFTY 10000000

struct CoinChangeHands {
    vector<long> min_changes_;

    CoinChangeHands(int N): min_changes_(N + 1, INFTY) {
        auto &tbl = min_changes_;
        vector<int> coins = {1, 2, 5};
        tbl[0] = 0;
        for (int i = 1; i <= N; ++i) {
            long curr = tbl[i];
            if (i >= 1) curr = min(curr, 1 + tbl[i-1]);
            if (i >= 2) curr = min(curr, 1 + tbl[i-2]);
            if (i >= 5) curr = min(curr, 1 + tbl[i-5]);
            tbl[i] = curr;
        }
    }
};

CoinChangeHands cx_hands(1007);

class Distributor {
    vector<long> nums;

public:
    Distributor(int n): nums(n) {}

    int run() {
        long min_val = INFTY;
        long max_val = 0;
        for (int i = 0; i < nums.size(); ++i) {
            cin >> nums[i];
            min_val = min(nums[i], min_val);
            max_val = max(nums[i], max_val);
        }
        long min_cnts = INFTY;
        // Adjust so that there is a chance for some number to be multiple
        // of the coin denominations, since the max is 5, set it to -5.
        for (int d = -5; d <= min_val; ++d) {
            long cnts = 0;
            for (auto a: nums) cnts += cx_hands.min_changes_[a - d];
            min_cnts = min(cnts, min_cnts);
        }
        return min_cnts;
    }
};

int main() {
    int T; cin >> T;
    for (; T > 0; --T) {
        int N; cin >> N;
        cout << Distributor(N).run() << endl;
    }
}

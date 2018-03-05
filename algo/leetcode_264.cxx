#include <iostream>
#include <vector>
#include <queue>

using namespace std;

class Solution {
    using int2 = pair<long long, int>;
public:
    int nthUglyNumber(int n) {
        if (n <= 0) return 0;
        if (n <= 1) return 1;

        vector<long long> tbl(n);
        tbl[0] = 1;
        int i2 = 0, i3 = 0, i5 = 0;
        for (int i = 1; i < n; ++i) {
            auto a2 = tbl[i2] * 2;
            auto a3 = tbl[i3] * 3;
            auto a5 = tbl[i5] * 5;
            // Tie breaking will happen all at once
            // All numbers are represented as 2^i * 3^j * 5^k.
            // The increment path will reach the same number
            // in i + j + k steps
            auto a_next = min(a2, min(a3, a5));
            if (a_next == a2) ++i2;
            if (a_next == a3) ++i3;
            if (a_next == a5) ++i5;
            tbl[i] = a_next;
        }
        return tbl[n - 1];
    }

    int nthUglyNumberWithQueue(int n) {
        priority_queue<int2, vector<int2>, greater<int2>> next_nums;
        next_nums.push(make_pair(1, 3));
        while (!next_nums.empty()) {
            auto num = get<0>(next_nums.top());
            int remaining_num_factors = get<1>(next_nums.top());
            next_nums.pop();
            if (--n == 0)  return num;
            switch (remaining_num_factors) {
            case 3:
                next_nums.push(make_pair(num * 2, 3));
            case 2:
                next_nums.push(make_pair(num * 3, 2));
            case 1:
                next_nums.push(make_pair(num * 5, 1));
            }
        }
        return -1;
    }
};

Solution sol;

void TEST(int n) { cout << sol.nthUglyNumber(n) << endl; }

int main() {
    TEST(11);
    TEST(1407);
}

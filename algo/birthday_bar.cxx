#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

int main() {
    int n; cin >> n;
    vector<int> nums(n);
    for (int i = 0; i < n; ++i) cin >> nums[i];
    int m, d; cin >> d >> m;

    // Rolling sum with bound check
    int i = 0, j = 1;
    int curr_sum = nums[0];
    int ways = 0;
    for (; j <= n; ++j) {
        for (; curr_sum > d && i < j; ++i) {
            curr_sum -= nums[i];
        }
        if (d == curr_sum && i + m == j) {
            ++ways;
        }
        if (j < n) curr_sum += nums[j];
    }
    cout << ways << endl;
    return 0;
}

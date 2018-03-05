/**
 * Find sliding window median
 */

#include <iostream>
#include <vector>
#include <set>
#include <cassert>

using namespace std;

class Solution {
public:
    vector<double> medianSlidingWindow(vector<int> &nums, int k) {
        multiset<double, greater<double>> lo_half_vals;
        multiset<double> hi_half_vals;
        vector<double> results;
        auto n = nums.size();
        for (int i = 0; i < n; ++i) {
            // Delete the element out of the k-sized window
            if (i >= k) {
                int prev = nums[i - k];
                auto it = lo_half_vals.find(prev);
                if (it != lo_half_vals.end()) {
                    lo_half_vals.erase(it);
                } else {
                    // Must use iterator to erase,
                    // or we will delete all occurences
                    it = hi_half_vals.find(prev);
                    hi_half_vals.erase(it);
                }
            }

            // Insert new element
            int curr = nums[i];
            auto lo_max = *lo_half_vals.begin();
            if (curr <= lo_max) {
                lo_half_vals.insert(curr);
            } else {
                hi_half_vals.insert(curr);
            }

            // Balance the two halfs
            if (lo_half_vals.size() + 2 <= hi_half_vals.size()) {
                auto top = hi_half_vals.begin();
                lo_half_vals.insert(*top);
                hi_half_vals.erase(top);
            } else if (hi_half_vals.size() + 2 <= lo_half_vals.size()) {
                auto top = lo_half_vals.begin();
                hi_half_vals.insert(*top);
                lo_half_vals.erase(top);
            }

            // Don't show for the initial values
            if (i + 1 < k) continue;

            if (0 == k % 2) {
                auto median = *lo_half_vals.begin() + *hi_half_vals.begin();
                results.push_back(median / 2.0);
                continue;
            }
            if (lo_half_vals.size() < hi_half_vals.size()) {
                results.push_back(*hi_half_vals.begin());
            } else {
                results.push_back(*lo_half_vals.begin());
            }
        }
        return results;
    }
};

Solution sol;

void TEST(vector<int> nums, int k) {
    auto res = sol.medianSlidingWindow(nums, k);
    for (auto a: res) cout << a << " ";
    cout << endl;
}

int main() {
    TEST({1,3,-1,-3,5,3,6,7}, 3);
    TEST({5,5,8,1,4,7,1,3,8,4}, 8);
}

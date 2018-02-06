/**
 * Burst ballons
 */

#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    int maxCoins(vector<int> &nums) {
        if (nums.empty()) return 0;
        // Add boundary values 1 to both sides
        nums.insert(nums.begin(), 1); nums.push_back(1);
        int n = nums.size();
        // Boundary condition: for the interval (i, i+1) => 0
        vector<vector<int>> tbl_intv_max_val(n, vector<int>(n, 0));
        for (int intv_size = 3; intv_size <= n; ++intv_size) {
            for (int i = 0; i + intv_size <= n; ++i) {
                int j = i + intv_size - 1;
                int curr_max = 0;
                for (int k = i + 1; k < j; ++k) {
                    int pref = tbl_intv_max_val[i][k];
                    int post = tbl_intv_max_val[k][j];
                    int curr = nums[i] * nums[k] * nums[j];
                    curr_max = max(curr_max, pref + curr + post);
                }
                tbl_intv_max_val[i][j] = curr_max;
            }
        }
        return tbl_intv_max_val[0][n - 1];
    }
};

Solution sol;

void TEST(vector<int> nums, int expected) {
    int res = sol.maxCoins(nums);
    if (res != expected)
        cout << "FAIL: expect " << expected << " but got " << res << endl;
    else
        cout << "PASS" << endl;
}

int main() {
    TEST({3,1,5,8}, 167);
    TEST({35,16,83,87,84,59,48,41}, 1583373);
}

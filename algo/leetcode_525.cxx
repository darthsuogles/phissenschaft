#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;

class Solution {
public:
    /**
     * Keeping a running count of first encounter of particular values
     */
    int findMaxLength(vector<int>& nums) {
        if (nums.empty()) return 0;
        unordered_map<int, int> init_pos_with_cnts;
        init_pos_with_cnts[0] = -1;
        int one_cnts = 0;
        int max_len = 0;
        for (auto i = 0; i < nums.size(); ++i) {
            auto val = nums[i];
            one_cnts += val + (val - 1);
            auto maybe_init_pos = init_pos_with_cnts.find(one_cnts);
            if (maybe_init_pos != init_pos_with_cnts.end()) {
                int len = i - maybe_init_pos->second;
                max_len = max(len, max_len);
            } else {
                init_pos_with_cnts[one_cnts] = i;
            }
        }
        return max_len;
    }
};

Solution sol;

void TEST(vector<int> nums, const int tgt) {
    auto res = sol.findMaxLength(nums);
    if (res != tgt) {
        cerr << "FAIL: " << res << " != " << tgt << endl;
    } else {
        cout << "PASS" << endl;
    }
}

int main() {
    TEST({0, 1}, 2);
    TEST({0, 1, 0}, 2);
    TEST({0, 0, 0}, 0);
    TEST({0, 1, 0, 1}, 4);
}

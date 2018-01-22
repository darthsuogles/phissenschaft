#include <iostream>
#include <vector>

using namespace std;

int maxProduct(vector<int>& nums) {
    long res = INT_MIN;
    if (nums.empty()) return res;
    res = nums[0];
    
    long curr_pos = 1, curr_neg = 1;
    for (auto a: nums) {
        if (a == 0) {
            curr_pos = curr_neg = 1;
            res = max(res, 0L);
            continue;
        }
        if (a > 0) {
            curr_pos *= a;
            curr_neg *= a;
        } else {
            if (curr_neg > 0) {
                curr_pos = 1; curr_neg *= a;
                continue;
            }
            int tmp = curr_pos;
            curr_pos = curr_neg * a;            
            curr_neg = tmp * a;
        }
        res = max(res, curr_pos);
    }
    return res;
}

void TEST(vector<int> nums, int tgt) {
    int res = maxProduct(nums);
    if (res == tgt) {
        cout << "OK" << endl;
    } else {
        cout << "ERROR " << res << " != " << tgt << endl;
    }
}

int main() {
    TEST({2,3,-2,4}, 6);
    TEST({1}, 1);
    TEST({-1}, -1);
    TEST({0}, 0);
    TEST({2,3,0,4}, 6);
    TEST({2,3,0,2,4,0}, 8);
    TEST({-2,0,-1}, 0);
    TEST({3,-1,4}, 4);
    TEST({7,-2,-4}, 56);
}

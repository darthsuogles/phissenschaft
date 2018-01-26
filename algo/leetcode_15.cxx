/**
 * 3Sum: find triplets that sum to one
 */

#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

vector< vector<int> > threeSum(vector<int> &nums) {
    vector< vector<int> > triplets;
    if (nums.empty()) return triplets;
    int n = nums.size();
    int target = 0;
    sort(nums.begin(), nums.end());
    for (auto idx = 0; idx + 2 < n; ++idx) {
        if (idx > 0 && nums[idx-1] == nums[idx]) 
            continue;  // skip duplicates trailings
        int i = idx + 1, j = n - 1;
        int curr = nums[idx];
        int curr_target = target - curr;
        while (i < j) {
            int a = nums[i], b = nums[j];
            if (a + b == curr_target) {
                triplets.push_back({curr, a, b});
                for (; i < j && nums[i] == a; ++i);
                for (; i < j && nums[j] == b; --j);
                continue;
            }             
            if ( a + b < curr_target ) ++i; else ++j;
        }
    }
    return triplets;
}


void TEST(vector<int> nums) {
    auto res = threeSum(nums);
    for (auto v: res) 
        printf("%+d %+d %+d\n", v[0], v[1], v[2]);
    cout << endl;
}


int main() {
    TEST({-1, 0, 1, 2, -1, -4});
    TEST({-4,-2,-2,-2,0,1,2,2,2,3,3,4,4,6,6});
}

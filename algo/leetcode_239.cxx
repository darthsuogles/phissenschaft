#include <iostream>
#include <vector>
#include <list>

using namespace std;

vector<int> maxSlidingWindow(vector<int>& nums, int k) {
    vector<int> res;
    if (nums.empty()) return res;
    list<int> max_val_inds;
    for (int i = 0; i < nums.size(); ++i) {
        auto a = nums[i];
        while (!max_val_inds.empty()) {
            if (max_val_inds.front() + k > i) break;
            max_val_inds.pop_front();
        }
        while (!max_val_inds.empty()) {
            auto b = nums[max_val_inds.back()];
            if (b >= a) break;
            max_val_inds.pop_back();
        }
        max_val_inds.push_back(i);
        if (i + 1 >= k)
            res.push_back(nums[max_val_inds.front()]);
    }
    return res;
}

int main() {
    vector<int> nums = {1,3,-1,-3,5,3,6,7};
    auto res = maxSlidingWindow(nums, 3);
    for (auto a: res)
        cout << a << endl;
}

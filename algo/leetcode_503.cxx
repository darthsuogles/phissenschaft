#include <iostream>
#include <vector>
#include <stack>

using namespace std;

vector<int> nextGreaterElements(vector<int> &nums) {
    if (nums.empty()) return {};
    int len = nums.size();
    vector<int> next_gt(len, -1);
    stack<int> dec_stack;  // decreasing order
    for (int iter = 0; iter < 2; ++iter) {
        for (int i = 0; i < len; ++i) {
            int a = nums[i];
            while (! dec_stack.empty()) {
                int j = dec_stack.top();
                int b = nums[j];
                if (b >= a) break;            
                dec_stack.pop();
                next_gt[j] = a;
            }
            dec_stack.push(i);
        }
    }
    return next_gt;
}

int main() {
    vector<int> nums = {1,2,1};
    auto res = nextGreaterElements(nums);
    for (auto a: res) cout << a << " ";
    cout << endl;
}

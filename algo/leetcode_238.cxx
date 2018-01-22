#include <iostream>
#include <vector>

using namespace std;

vector<int> productExceptSelf(vector<int> &nums) {
    vector<int> res;
    if (nums.empty()) return res;
    int n = nums.size();

    vector<int> prod_avant, prod_apres;
    long curr = 1;
    for (auto a : nums) {
        prod_avant.push_back(curr); curr *= a;
    }
    curr = 1;
    for (auto it = nums.rbegin(); it != nums.rend(); ++it) {
        prod_apres.push_back(curr); curr *= *it;
    }

    for (int i = 0; i < n; ++i) {
        res.push_back(prod_avant[i] * prod_apres[n-i-1]);
    }
    return res;
}

#define TEST(...) {                           \
    auto nums = vector<int>{__VA_ARGS__};     \
    auto res = productExceptSelf(nums);       \
    for (auto a : res) cout << a << " "; cout << endl; }

int main() {
    TEST(1,2,3,4);
}

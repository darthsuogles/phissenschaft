#include <iostream>
#include <vector>

using namespace std;

// It really means, add 1 to everyone for n rounds
// then subtracting 1s for n rounds for single elements
int minMoves(vector<int>& nums) {
    if (nums.size() <= 1) return 0;
    int v_min = nums[0];
    for (auto a: nums) v_min = min(v_min, a);
    long diff = 0;
    for (auto a: nums) diff += a - v_min;
    return (int) diff;
}

#define TEST(...) {                             \
        auto nums = vector<int> {__VA_ARGS__};  \
        cout << minMoves(nums) << endl; }

int main() {
    TEST(1,2,3);
}

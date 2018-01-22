#include <iostream>
#include <vector>

using namespace std;

void moveZeroes(vector<int> &nums) {
    int i = 0, j = i;
    for (; i < nums.size(); ++i) {
        if (0 != nums[i])
            nums[j++] = nums[i];
    }
    for (; j < nums.size(); ++j) {
        nums[j] = 0;
    }
}

int main() {
    auto nums = vector<int> {0, 1, 0, 3, 12};
    moveZeroes(nums);
    for (auto a : nums) cout << a << " "; cout << endl;
}

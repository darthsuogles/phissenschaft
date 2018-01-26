/**
 * K-diff pairs
 */

#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

int findPairs(vector<int> &nums, int k) {
    if (nums.empty()) return 0;
    sort(nums.begin(), nums.end());
    //for (auto a: nums) cout << a << endl;
    int i = 0;
    auto len = nums.size();
    int cnts = 0;
    while (i < len) {
        int a = nums[i];
        int j = i + 1;
        for (; j < len; ++j) {
            int b = nums[j];
            if (a + k <= b) {
                cnts += int(a + k == b);
                break;
            }
        }
        for (j = i + 1; j < len; ++j)
            if (a != nums[j]) break;
        i = j;
    }
    return cnts;
}

void TEST(vector<int> nums, int k, int expected) {
    int res = findPairs(nums, k);
    if (res != expected)
        cout << "Error " << res << " but expect " << expected;
    else
        cout << "Ok" << endl;
}

int main() {
    TEST({3, 1, 4, 1, 5}, 2, 2);
    TEST({1, 2, 3, 4, 5}, 1, 4);
    TEST({1, 3, 1, 5, 4}, 0, 1);
}

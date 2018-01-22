#include <iostream>
#include <vector>

using namespace std;

int thirdMax(vector<int>& nums) {
    vector<long> trois(3, LONG_MIN);
    for (auto a: nums) {
        int idx = -1;
        bool dup = false;
        for (int r = 0; r < 3; ++r, ++idx) {
            if (a == trois[r]) dup = true;
            if (a <= trois[r]) break;
        }
        if (dup || -1 == idx) continue;
        for (int r = 0; r < idx; ++r) 
            trois[r] = trois[r+1];
        trois[idx] = a;
    }
    if (trois[0] > LONG_MIN) return trois[0];
    return trois[2];
}

void TEST(vector<int> nums, int tgt) {
    int res = thirdMax(nums);
    if (res == tgt) 
        cout << "OK" << endl;
    else
        cout << "ERROR " << res << " != " << tgt << endl;
}

int main() {
    TEST({3, 2, 1}, 1);
    TEST({1, 2}, 2);
    TEST({2, 2, 3, 1}, 1);
    TEST({1,2,-2147483648}, -2147483648);
}

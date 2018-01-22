#include <iostream>
#include <vector>

using namespace std;

int findDuplicateBS(vector<int>& nums) {
    if (nums.empty()) return INT_MIN;
    auto n = nums.size() - 1;
    int vmin = 1, vmax = n;
    while (vmin + 1 < vmax) {
        int mid = (vmin + vmax) / 2;
        int cnt = 0;
        for (auto a: nums) 
            if (vmin <= a && a <= mid) ++cnt;
        int cnt_exp = mid - vmin + 1;
        if (cnt > cnt_exp) 
            vmax = mid;
        else
            vmin = mid + 1;
    }
    int cnt = 0;
    for (auto a: nums) 
        if (a == vmin) ++cnt;
    if (cnt > 1) 
        return vmin;
    else
        return vmax;
}

int findDuplicate(vector<int>& nums) {
    int n = nums.size();
    int i = n, j = n;
#define F(I) nums[(I) - 1]
    do { i = F(i); j = F(F(j)); } while (i != j);
    for (j = n; i != j; i = F(i), j = F(j));
    return j;
}

void TEST(vector<int> nums, int tgt) {
    int res = findDuplicate(nums);
    if (res != tgt)
        cout << "ERROR: " << res << " != " << tgt;
    else
        cout << "OK";
    cout << endl;            
}

int main() {
    TEST({1,1,2,3,4,5,6}, 1);
    TEST({1,2,1,1,3,5,6}, 1);
    TEST({1,2,2,2,3,5,6}, 2);
    TEST({1,2,4,2,3,5,6}, 2);
    TEST({2,1,2}, 2);
    TEST({2,1,1}, 1);
    TEST({2,1,1,1,4}, 1);
}

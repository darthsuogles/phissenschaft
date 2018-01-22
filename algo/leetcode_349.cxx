#include <iostream>
#include <vector>
#include <unordered_set>

using namespace std;

vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
    unordered_set<int> tbl(nums1.begin(), nums1.end());
    unordered_set<int> res;
    for (auto a : nums2)
        if (tbl.count(a) > 0) res.insert(a);
    
    return vector<int>(res.begin(), res.end());
}

void TEST(vector<int> arr1, vector<int> arr2) { 
    auto res = intersection((arr1), (arr2)); 
    for (auto a : res) cout << a << " "; cout << endl; 
}

int main() {
    TEST({1,2,2,1}, {2,2});
}

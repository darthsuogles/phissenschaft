#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>

using namespace std;

class BigInt {
    string repr_;
public:
    friend istream& operator >> (istream &is, BigInt &num) {
        getline(is, num.repr_);
        return is;
    }
    friend ostream& operator << (ostream &os, BigInt &num) {
        return os << num.repr_;
    }
    static bool compare(const BigInt &a, const BigInt &b) {
        auto a_len = a.repr_.size();
        auto b_len = b.repr_.size();
        if (a_len != b_len) return a_len < b_len;
        return a.repr_ < b.repr_;
    }
};

int main() {
    int n; cin >> n;
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
    vector<BigInt> nums(n);
    for (int i = 0; i < n; ++i) {
        cin >> nums[i];
    }
    sort(nums.begin(), nums.end(), BigInt::compare);
    for (auto num: nums) {
        cout << num << endl;
    }
}

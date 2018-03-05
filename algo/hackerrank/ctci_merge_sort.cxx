#include <vector>
#include <iostream>

using namespace std;

long count_inversions(vector<int>::iterator arr_begin, vector<int>::iterator arr_end) {
    auto n = distance(arr_begin, arr_end);
    if (n < 2) return 0;
    auto arr_mid = arr_begin + n / 2;
    int prefix_invs = count_inversions(arr_begin, arr_mid);
    int suffix_invs = count_inversions(arr_mid, arr_end);

    vector<int> merged;
    auto it = arr_begin;
    auto jt = arr_mid;
    long merged_invs = 0;
    while (it != arr_mid && jt != arr_end) {
        if (*jt < *it) {
            merged.push_back(*jt++);
            merged_invs += distance(it, arr_mid);
        } else {
            merged.push_back(*it++);
        }
    }
    while (it != arr_mid) {
        merged.push_back(*it++);
    }
    while (jt != arr_end) {
        merged.push_back(*jt++);
    }
    for (auto it = arr_begin, jt = merged.begin();
         it != arr_end && jt != merged.end(); *it++ = *jt++);

    return merged_invs + prefix_invs + suffix_invs;
}

int main() {
    int d; cin >> d;
    for (; d > 0; --d) {
        int n; cin >> n;
        vector<int> nums(n);
        for (int i = 0; i < n; ++i) {
            cin >> nums[i];
        }
        cout << count_inversions(nums.begin(), nums.end()) << endl;
    }
}

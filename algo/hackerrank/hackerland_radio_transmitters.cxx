#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int main() {
    int n, k; cin >> n >> k;
    if (1 == n) {
        cout << 1 << endl;
        return 0;
    }

    int cover_size = k + k;
    vector<int> nums(n);
    for (int i = 0; i < n; ++i) {
        cin >> nums[i];
    }
    sort(nums.begin(), nums.end());

    int left_edge = nums[0];
    int right_edge = -1;
    int cnts = 0;
    for (int i = 1; i < n; ++i) {
        if (nums[i] - left_edge > k) {
            int install_site = nums[i-1];
            for (; i < n; ++i) {
                if (nums[i] - install_site > k) {
                    left_edge = nums[i];
                    right_edge = nums[i-1];
                    ++cnts;  // increment counts when we've closed a site range
                    break;
                }
            }
        }
    }
    if (right_edge != nums[n-1]) ++cnts;
    cout << cnts << endl;
}

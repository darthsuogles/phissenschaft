#include <iostream>
#include <vector>
#include <set>
#include <cassert>

using namespace std;

multiset<int, greater<int>> lo_half_max;
multiset<int, less<int>> hi_half_min;

void balance_heaps() {
    if (lo_half_max.size() + 1 < hi_half_min.size()) {
        auto it = hi_half_min.begin();
        lo_half_max.insert(*it);
        hi_half_min.erase(it);
    } else if (hi_half_min.size() + 1 < lo_half_max.size()) {
        auto it = lo_half_max.begin();
        hi_half_min.insert(*it);
        lo_half_max.erase(it);
    }
}

void show_heaps() {
    cout << "lo_half_max: ";
    for (auto a: lo_half_max) cout << a << " ";
    cout << endl;
    cout << "hi_half_min: ";
    for (auto a: hi_half_min) cout << a << " ";
    cout << endl;

}

int main() {
    int n, k; cin >> n >> k;
    vector<int> nums(n);
    for (int i = 0; i < n; ++i) {
        cin >> nums[i];
    }

    lo_half_max.insert(nums[0]);
    for (int i = 1; i < k; ++i) {
        auto lo_max = *lo_half_max.begin();
        if (nums[i] <= lo_max) {
            lo_half_max.insert(nums[i]);
        } else {
            hi_half_min.insert(nums[i]);
        }
        balance_heaps();
    }

    int num_alerts = 0;
    for (int i = k; i < n; ++i) {
        //show_heaps();
        // Check median
        int med2;
        if (1 == k % 2) {
            assert(lo_half_max.size() + 1 >= hi_half_min.size());
            assert(hi_half_min.size() + 1 >= lo_half_max.size());

            if (lo_half_max.size() > hi_half_min.size()) {
                med2 = *lo_half_max.begin();
            } else {
                med2 = *hi_half_min.begin();
            }
            med2 *= 2;
        } else {
            assert(lo_half_max.size() == hi_half_min.size());

            med2 = *lo_half_max.begin() + *hi_half_min.begin();
        }

        //cout << k << ": " << median << endl;
        if (med2 <= nums[i]) ++num_alerts;

        // Remove value fallen out of the window
        int rm_val = nums[i - k];
        auto it = lo_half_max.find(rm_val);
        if (it != lo_half_max.end()) {
            lo_half_max.erase(it);
        } else {
            hi_half_min.erase(hi_half_min.find(rm_val));
        }
        balance_heaps();

        // Add current value
        int add_val = nums[i];
        auto lo_max = *lo_half_max.begin();
        if (add_val <= lo_max) {
            lo_half_max.insert(add_val);
        } else {
            hi_half_min.insert(add_val);
        }
        balance_heaps();
    }
    cout << num_alerts << endl;
}

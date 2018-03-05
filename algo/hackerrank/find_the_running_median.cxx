#include <iostream>
#include <set>

using namespace std;

int main() {
    int n; cin >> n;

    multiset<int, greater<int>> lo_half_desc; // largest on top
    multiset<int> hi_half_incr; // smallest on top

    for (int i = 0; i < n; ++i) {
        int a; cin >> a;
        if (a <= *lo_half_desc.begin()) {
            lo_half_desc.insert(a);
        } else {
            hi_half_incr.insert(a);
        }
        // Balance the two halfs
        if (lo_half_desc.size() + 1 < hi_half_incr.size()) {
            auto it = hi_half_incr.begin();
            lo_half_desc.insert(*it);
            hi_half_incr.erase(it);
        } else if (hi_half_incr.size() + 1 < lo_half_desc.size()) {
            auto it = lo_half_desc.begin();
            hi_half_incr.insert(*it);
            lo_half_desc.erase(it);
        }
        // Check the median
        float median;
        if (lo_half_desc.size() < hi_half_incr.size()) {
            median = *hi_half_incr.begin();
        } else if (lo_half_desc.size() > hi_half_incr.size()) {
            median = *lo_half_desc.begin();
        } else {
            float val = *lo_half_desc.begin() + *hi_half_incr.begin();
            median = val / 2.0;
        }
        printf("%.1f\n", median);
    }
}

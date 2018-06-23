#include <vector>
#include <iostream>
#include <algorithm>
#include <utility>
#include <map>

using namespace std;

using integer = long;

integer get_num_swaps(vector<integer> raw_vals) {
    auto n = raw_vals.size();
    // Array only contains distinct values
    map<integer, integer> val2idx;
    for (auto i = 0; i < n; ++i) {
        val2idx[raw_vals[i]] = i;
    }

    integer num_swaps = 0;
    int i = -1;
    for (auto kv: val2idx) {
        ++i;
        int a = get<0>(kv);
        int j = get<1>(kv);
        int b = raw_vals[i];
        if (a == b) continue;

        val2idx[b] = j;
        swap(raw_vals[i], raw_vals[j]);
        ++num_swaps;
    }
    return num_swaps;
}

int main() {
    int n; cin >> n;
    vector<integer> idx2val(n);
    for (int i = 0; i < n; ++i) {
        integer a; cin >> a;
        idx2val[i] = a;
    }
    auto min_swaps = get_num_swaps(idx2val);
    reverse(idx2val.begin(), idx2val.end());
    min_swaps = min(get_num_swaps(idx2val), min_swaps);
    cout << min_swaps << endl;
    return 0;
}

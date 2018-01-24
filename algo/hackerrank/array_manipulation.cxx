/**
 * Array manipulation
 *
 * Increment elements in a range and return the largest elememt
 */

#include <iostream>
#include <utility>
#include <memory>
#include <algorithm>
#include <map>
#include <vector>

using namespace std;

struct Record {
    int val;
    bool used;
    Record(int val): val(val), used(false) {}
};

using idx_rec_t = pair<int, shared_ptr<Record>>;

void merge_interval() {
    int m, n;
    cin >> n >> m;
    vector<idx_rec_t> init_irec;
    vector<idx_rec_t> fini_jrec;
    int i, j, val;
    while (cin >> i >> j >> val) {
        auto rec = make_shared<Record>(val);
        init_irec.push_back(make_pair(i, rec));
        fini_jrec.push_back(make_pair(j, rec));
    }
    // Sort the intervals by their ending
    auto comp = [](idx_rec_t &r1, idx_rec_t &r2) { return get<0>(r1) < get<0>(r2); };
    sort(init_irec.begin(), init_irec.end(), comp);
    sort(fini_jrec.begin(), fini_jrec.end(), comp);

    int max_val = 0;
    for (auto jrec: fini_jrec) {
        int j = get<0>(jrec);
        int curr_val = 0;
        for (auto irec: init_irec) {
            if (get<0>(irec) > j) break;
            auto rec = get<1>(irec);
            if (rec->used) continue;
            curr_val += rec->val;
        }
        max_val = max(max_val, curr_val);
        get<1>(jrec)->used = true;
    }
    cout << max_val << endl;
}

// Using the "difference array" approach:
// Like tracking the "gradient" of the curve
// This is also true as "+" is superpositional.
// https://wcipeg.com/wiki/Prefix_sum_array_and_difference_array
void incr_decr_dense() {
    int n, m;
    cin >> n >> m;
    vector<long> point_cnts(n + 2, 0);
    int i, j, val;
    while (cin >> i >> j >> val) {
        point_cnts[i] += val;
        point_cnts[j + 1] -= val;
    }
    long curr_sum = 0L;
    long max_val = 0L;
    for (int i = 1; i <= n; ++i) {
        curr_sum += point_cnts[i];
        max_val = max(curr_sum, max_val);
    }
    cout << max_val << endl;
}

void incr_decr_sparse() {
    int n, m;
    cin >> n >> m;
    map<int, long> sparse_pcnts;
    int i, j, val;
    while (cin >> i >> j >> val) {
        sparse_pcnts[i] += val;
        sparse_pcnts[j + 1] -= val;
    }
    long trace_sum = 0L;
    long max_val = 0L;
    for (auto ival: sparse_pcnts) {
        trace_sum += get<1>(ival);
        max_val = max(max_val, trace_sum);
    }
    cout << max_val << endl;
}

int main() {
    incr_decr_sparse();
}

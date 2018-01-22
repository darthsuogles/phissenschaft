#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>

using namespace std;

typedef pair<int, int> kv_t;

struct _op_ord {
    bool operator() (const kv_t& a, const kv_t &b) const {
        return a.second < b.second;
    }
};


// Using a max-heap 
vector<int> topKFrequentHeap(vector<int> &nums, int k) {
    vector<int> res;
    if (0 == k) return res;
    unordered_map<int, int> cntr;
    for (auto a : nums) ++cntr[a];
    priority_queue<kv_t, vector<kv_t>, _op_ord> pq(cntr.begin(), cntr.end());
    while (!pq.empty() && k-- > 0) {
        res.push_back(pq.top().first); pq.pop();
    }
    return res;
}

vector<int> topKFrequent(vector<int> &nums, int k) {
    vector<int> res;
    if (0 == k) return res;
    unordered_map<int, int> cntr;
    for (auto a : nums) ++cntr[a];
    // Observe that there are at most n different counts
    vector< vector<int> > tbl; tbl.resize(nums.size());
    for (auto &kv : cntr) {
        tbl[kv.second - 1].push_back(kv.first);
    }    
    for (auto it = tbl.rbegin(); it != tbl.rend() && k > 0; ++it) {
        if (it->empty()) continue;
        for (auto jt = it->begin(); jt != it->end() && k-- > 0; ++jt) 
            res.push_back(*jt);
    }
    return res;
}

void TEST(vector<int> nums, int k) {
    auto res = topKFrequent(nums, k);
    for (auto a : res) cout << a << " ";
    cout << endl;
}

int main() {
    TEST({1,1,1,2,2,3}, 2);
}

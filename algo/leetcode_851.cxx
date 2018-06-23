#include <iostream>
#include <tuple>
#include <vector>
#include <utility>

using namespace std;


class Solution {
    using adj_list_t = vector<vector<int>>;
    using cache_t = vector<pair<int, int>>;
    static constexpr pair<int, int> INVALID = make_pair(-1, -1);

public:
    vector<int> loudAndRich(vector<vector<int>>& richer, vector<int>& quiet) {
        int num_verts = quiet.size();
        adj_list_t neighbors(num_verts);
        for (auto uv: richer) {
            neighbors[uv[1]].push_back(uv[0]);
        }
        vector<int> res;
        cache_t cache(num_verts, INVALID);
        for (int u = 0; u < num_verts; ++u) {
            auto vert_val = dfs_find_min(u, neighbors, quiet, cache);
            res.push_back(get<0>(vert_val));
        }
        return res;
    }

private:
    pair<int, int> dfs_find_min(int u, adj_list_t &neighbors, vector<int> &vals, cache_t &cache) {
        if (INVALID != cache[u]) return cache[u];
        int min_val = vals[u];
        int min_vert = u;
        for (auto v: neighbors[u]) {
            auto vert_val = dfs_find_min(v, neighbors, vals, cache);
            if (get<1>(vert_val) < min_val) {
                min_val = get<1>(vert_val);
                min_vert = get<0>(vert_val);
            }
        }
        return cache[u] = make_pair(min_vert, min_val);
    }
};

Solution sol;

void TEST(vector<vector<int>> richer, vector<int> quiet, vector<int> tgt) {
    auto res = sol.loudAndRich(richer, quiet);

    int num_failed = 0;
    for (int i = 0; i < tgt.size(); ++i) {
        if (res[i] == tgt[i]) continue;
        cerr << "FAIL: [" << i << "] " << res[i] << " != " << tgt[i] << endl;
        ++num_failed;
    }
    if (0 == num_failed)
        cout << "PASS" << endl;
}

int main() {
    TEST({{1,0},{2,1},{3,1},{3,7},{4,3},{5,3},{6,3}},
         {3,2,5,4,6,1,7,0},
         {5,5,2,5,4,5,6,7});
}

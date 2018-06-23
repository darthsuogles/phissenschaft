// TODO: https://www.hackerrank.com/challenges/kingdom-division/editorial

#include <iostream>
#include <vector>
#include <cassert>

using namespace std;

using adj_list_t = vector<vector<int>>;
using integer = long long;

class FindWays {
    const adj_list_t adj_list;
    vector<vector<vector<integer>>> sub_tbl_;
    vector<int> parent_;
    static const integer MOD = 1e9 + 7;
    vector<int> visit_order;

public:
    FindWays(const adj_list_t &neighbors)
        : adj_list(neighbors),
          sub_tbl_(2, vector<vector<integer>>(2, vector<integer>(neighbors.size(), -1))),
          parent_(neighbors.size(), -1) {
        // First root the tree at 0
        set_parents(0);
    }

    void set_parents(int u) {
        for (int v: adj_list[u]) {
            if (0 == v || -1 != parent_[v]) continue;
            parent_[v] = u;
            set_parents(v);
        }
        visit_order.push_back(u);
    }

    inline bool is_leaf(int u) {
        if (0 == u) return false; // root
        if (adj_list[u].size() > 1) return false;
        return true;
    }


    integer find_ways() {
        // Set initial conditions
        for (int u = 1; u < adj_list.size(); ++u) {
            if (!is_leaf(u)) continue;
            for (bool black: {true, false})
                for (bool safe: {true, false})
                    sub_tbl_[black][safe][u] = 1;
        }

        // Reverse topological ordering of the tree
        for (int u: visit_order) {
            if (is_leaf(u)) continue;
            bool has_leaf = false;
            for (int v: adj_list[u]) {
                has_leaf = is_leaf(v);
                if (has_leaf) break;
            }

            for (bool black: {true, false}) {
                for (bool safe: {true, false}) {
                    bool must_adjust = !(safe || has_leaf);
                    integer ways = 1;
                    integer sub = 1;
                    for (int v: adj_list[u]) {
                        if ((v == parent_[u]) || is_leaf(v)) continue;
                         integer cnts_black = (sub_tbl_[black][true][v] * ways) % MOD;
                        integer cnts_white = (sub_tbl_[!black][false][v] * ways) % MOD;
                        ways = (cnts_black + cnts_white) % MOD;
                        if (must_adjust)
                            sub = (sub * sub_tbl_[!black][false][v]) % MOD;
                    }
                    if (must_adjust) {
                        integer pos_sub = MOD - (sub % MOD);
                        ways = (ways + pos_sub) % MOD;
                    }
                    sub_tbl_[black][safe][u] = ways % MOD;
                }
            }
        }
        const bool black = true;
        const bool safe = true;
        return (sub_tbl_[black][!safe][0] + sub_tbl_[!black][!safe][0]) % MOD;
    }

};

int main() {
    int n; cin >> n; // number of nodes in the tree
    adj_list_t adj_list(n);
    for (int e = 0; e + 1 < n; ++e) {
        // Nodes of the graph are labeled with 1-based index
        int u, v; cin >> u >> v; --u; --v;
        adj_list[u].push_back(v);
        adj_list[v].push_back(u);
    }
    cout << FindWays(adj_list).find_ways() << endl;
}

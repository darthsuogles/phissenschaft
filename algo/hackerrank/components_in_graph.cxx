/**
 * https://www.hackerrank.com/challenges/components-in-graph/problem
 */

#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <utility>
#include <algorithm>

using namespace std;

class UnionFind {
    const int N;
    vector<int> parent;
    vector<int> rank;

public:
    UnionFind(int N): parent(vector<int>(N)), rank(vector<int>(N)), N(N) {
        for (int i = 0; i <= N; ++i) {
            parent[i] = i;
            rank[i] = 0;
        }
    }

    pair<int, int> count() {
        vector<int> counts(N + 1, 0);
        for (int u = 1; u <= N; ++u) {
            ++counts[find_parent(u)];
        }
        int min_cnts = N, max_cnts = 0;
        for (int cnt: counts) {
            if (cnt <= 1) continue;
            min_cnts = min(cnt, min_cnts);
            max_cnts = max(cnt, max_cnts);
        }
        return make_pair(min_cnts, max_cnts);
    }

    int find_parent(int u) {
        int pu = parent[u];
        if (pu != parent[pu])
            pu = find_parent(pu);
        return parent[u] = pu;
    }

    void merge(int u, int v) {
        int pu = find_parent(u);
        int pu_rank = rank[pu];
        int pv = find_parent(v);
        int pv_rank = rank[pv];
        if (pu_rank < pv_rank) {
            parent[pu] = pv;
        } else if (pv_rank < pu_rank) {
            parent[pv] = pu;
        } else {
            parent[pv] = pu;
            ++rank[pu];
        }
    }
};

int main() {
    int N;
    cin >> N;
    int u, v;
    vector<pair<int, int>> edges;
    while (cin >> u >> v) {
        edges.push_back(make_pair(u, v));
    }

    UnionFind dset(2 * N);
    for (auto e: edges) {
        int u = get<0>(e), v = get<1>(e);
        dset.merge(u, v);
    }
    auto min_max = dset.count();
    cout << get<0>(min_max) << " " << get<1>(min_max) << endl;
}

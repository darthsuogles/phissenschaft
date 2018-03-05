#include <iostream>
#include <vector>

using namespace std;

using integer = long long;

void dfs_cc(int u, const vector<vector<int>> &adj_list, vector<bool> &visited) {
    visited[u] = true;
    for (auto v: adj_list[u]) {
        if (visited[v]) continue;
        dfs_cc(v, adj_list, visited);
    }
}

integer roadsAndLibraries(int n, int c_lib, int c_road, const vector<vector<int>> &adj_list) {
    // First build a spanning forest to find out the number connected components, n_cc
    // This gives the minimum number of libraries to be restored.
    // Then the optimal number of libraries can be found via minimizing
    //   x * c_lib + (n - x) * c_road => x * (c_lib - c_road) + n * c_road
    // where n_cc <= x <= n, i.e., x is the desired number of connected components.

    vector<bool> visited(n, false);
    integer n_cc = 0;
    for (int city = 0; city < n; ++city) {
        if (visited[city]) continue;
        ++n_cc;
        dfs_cc(city, adj_list, visited);
    }
    integer x = (c_lib >= c_road) ? n_cc : n;
    return x * static_cast<integer>(c_lib) + (n - x) * static_cast<integer>(c_road);
}

int main() {
    int q;
    cin >> q;
    for(int a0 = 0; a0 < q; a0++){
        int n;
        int m;
        int c_lib;
        int c_road;
        cin >> n >> m >> c_lib >> c_road;
        vector<vector<int>> adj_list(n);
        for (int ei = 0; ei < m; ++ei) {
            int u, v; cin >> u >> v;  // 1-based indexing
            --u; --v;
            adj_list[u].push_back(v);
            adj_list[v].push_back(u);
        }
        cout << roadsAndLibraries(n, c_lib, c_road, adj_list) << endl;
    }
    return 0;
}

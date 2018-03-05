#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>

using namespace std;

using adj_list_t = unordered_map<int, vector<int>>;
const int N = 100;
const int MAX_DIST = 107;

int get_min_steps(adj_list_t &adj_list, const vector<bool> &winning_pos) {
    // Dijastra
    vector<int> dist(N + 1, MAX_DIST);
    dist[1] = 0;
    multimap<int, int> queue;
    queue.insert(make_pair(0, 1));
    vector<bool> visited(N + 1, false);
    while (!queue.empty()) {
        auto it = queue.begin();
        int d = it->first, u = it->second;
        queue.erase(it);
        visited[u] = true;
        dist[u] = d;
        if (winning_pos[u]) return d;

        for (auto v: adj_list[u]) {
            if (visited[v]) continue;
            queue.insert(make_pair(d + 1, v));
        }
        while (!queue.empty()) {
            auto it = queue.begin();
            int u = it->second;
            if (!visited[u]) break;
            queue.erase(it);
        }
    }
    return dist[N];
}

int main() {
    int T; cin >> T;
    while (--T >= 0) {
        adj_list_t adj_list;
        vector<bool> winning_pos(N + 1, false);
        winning_pos[N] = true;
        vector<int> next_cell(N + 1, -1);
        for (int u = 1; u < N; ++u) next_cell[u] = u + 1;
        for (int t = 0; t < 2; ++t) {
            int num_tunnels; cin >> num_tunnels;
            for (int i = 0; i < num_tunnels; ++i) {
                int init, fini; cin >> init >> fini;
                if (N == fini) {
                    winning_pos[init] = true;
                    next_cell[init] = -1;
                } else {
                    next_cell[init] = min(N, fini + 1);
                }
            }
        }
        for (int u = 1; u <= N; ++u) {
            int u_next = next_cell[u];
            if (-1 == u_next) continue;
            // u_next = u "+" 1
            for (int v = u_next; v <= min(N, u_next + 5); ++v) {
                adj_list[u].push_back(v);
                if (winning_pos[v]) break;
            }
        }
        int min_steps = get_min_steps(adj_list, winning_pos);
        if (MAX_DIST == min_steps) cout << -1 << endl;
        else cout << min_steps << endl;
    }
}

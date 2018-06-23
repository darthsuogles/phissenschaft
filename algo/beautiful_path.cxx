#include <iostream>
#include <vector>
#include <queue>
#include <utility>

using namespace std;

using adj_list_t = vector<vector<pair<int, int>>>;

const int MAX_DIST = 1024;

// TODO: https://www.hackerrank.com/challenges/beautiful-path/editorial

int find_shortest_path(adj_list_t &adj_list, int src, int dst) {
    // We use a BFS approach
    const int N = adj_list.size();
    queue<pair<int, int>> next_nodes;
    vector<vector<bool>> seen(N, vector<bool>(1024, false));
    next_nodes.push(make_pair(src, 0));
    while (!next_nodes.empty()) {
        auto top = next_nodes.front();
        int u = get<0>(top), u_dist = get<1>(top);
        next_nodes.pop();
        for (auto vd: adj_list[u]) {
            int v = get<0>(vd), v_dist = get<1>(vd);
            int delta = u_dist | v_dist;
            if (seen[v][delta]) continue;
            seen[v][delta] = true;
            next_nodes.push(make_pair(v, delta));
        }
    }
    for (int d = 0; d < MAX_DIST; ++d) {
        if (seen[dst][d]) return d;
    }
    return -1;
}

int find_shortest_path_bellman_ford(adj_list_t &adj_list, int src, int dst) {
    // Bellman & Ford
    int N = adj_list.size();
    vector<int> tbl(N, MAX_DIST);
    tbl[src] = 0;
    int num_update = 0;
    while (true) {
        for (int u = 0; u < N; ++u) {
            for (auto vd: adj_list[u]) {
                int v = get<0>(vd), d = get<1>(vd);
                int new_dist = (d | tbl[u]);
                if (new_dist < tbl[v]) {
                    tbl[v] = new_dist;
                    ++num_update;
                }
            }
        }
        if (0 == num_update) break;
        num_update = 0;
    }
    return tbl[dst];
}

int find_shortest_path_dijkstra(adj_list_t &adj_list, int src, int dst) {
    // Dijkstra
    priority_queue<pair<int, int>,
                   vector<pair<int, int>>,
                   greater<pair<int, int>>> next_nodes;

    const int N = adj_list.size();
    vector<bool> visited(N, false);
    vector<int> distance(N, MAX_DIST);
    next_nodes.push(make_pair(0, src));
    while (!next_nodes.empty()) {
        auto dist_and_node = next_nodes.top();
        int u_dist = get<0>(dist_and_node), u_node = get<1>(dist_and_node);
        next_nodes.pop();
        visited[u_node] = true;
        distance[u_node] = u_dist;
        for (auto vd: adj_list[u_node]) {
            int v_dist = get<1>(vd), v_node = get<0>(vd);
            if (visited[v_node]) continue;
            next_nodes.push(make_pair(u_dist | v_dist, v_node));
        }
        while (!next_nodes.empty()) {
            if (!visited[get<1>(next_nodes.top())]) break;
            next_nodes.pop();
        }
    }
    return distance[dst];
}

int main() {
    int N, M; cin >> N >> M;
    adj_list_t adj_list(N);
    for (int i = 0; i < M; ++i) {
        int u, v, c;
        cin >> u >> v >> c;
        --u; --v;
        adj_list[u].push_back(make_pair(v, c));
        adj_list[v].push_back(make_pair(u, c));
    }
    int A, B; cin >> A >> B;
    --A; --B;
    cout << find_shortest_path(adj_list, A, B) << endl;
}

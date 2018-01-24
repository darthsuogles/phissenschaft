#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <queue>
#include <utility>

using namespace std;


int main() {
  /* Enter your code here. Read input from STDIN. Print output to STDOUT */   
  int T; cin >> T;
  while ( T-- > 0 ) {
    int N, M; cin >> N >> M;
    vector<int> dist_to_target(N+1, -1);
    vector< vector<int> > edge_list(N+1, vector<int>(0));
    for (int e = 0; e < M; ++e) {
      int u, v; cin >> u >> v;
      edge_list[u].push_back(v);
      edge_list[v].push_back(u);
    }
    int node; cin >> node;
    dist_to_target[node] = 0;
    queue< pair<int, int> > bfs_visit_queue;
    bfs_visit_queue.push(make_pair(node, 0));
    while ( ! bfs_visit_queue.empty() ) {
      auto curr = bfs_visit_queue.front(); 
      bfs_visit_queue.pop();
      int node = curr.first;
      int dist = curr.second;
      for (auto it = edge_list[node].begin(); it != edge_list[node].end(); ++it) {
	int next = *it;
	if ( -1 != dist_to_target[next] )
	  continue;
	int next_dist = dist + 6;
	dist_to_target[next] = next_dist;
	bfs_visit_queue.push(make_pair(next, next_dist));
      }
    }
    for (int u = 1; u <= N; ++u) {
      int dist = dist_to_target[u];
      if ( 0 != dist )
	cout << dist << " ";
    }
    cout << endl;
  }
  return 0;
}


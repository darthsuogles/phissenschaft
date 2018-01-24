/**
 * https://www.hackerrank.com/challenges/dijkstrashortreach
 */

#include <cmath>
#include <cstdio>
#include <climits>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <algorithm>
#include <queue>
#include <utility>

using namespace std;

struct DistLEQ {
  bool operator() (const pair<int, int> &p1, const pair<int, int> &p2) const {
    return p1.second < p2.second;
  }
};

int main() {
  /* Enter your code here. Read input from STDIN. Print output to STDOUT */
  int T; cin >> T;
  while ( T-- > 0 ) {
    int N, M; cin >> N >> M;
    vector< unordered_map<int, int> > weighted_edge_list(N+1);
    for (int e = 0; e < M; ++e) {
      int u, v, dist; cin >> u >> v >> dist;
      int min_dist = dist;
      if ( weighted_edge_list[u].count(v) > 0 ) 
	min_dist = min(weighted_edge_list[u][v], dist);
      weighted_edge_list[u][v] = weighted_edge_list[v][u] = min_dist;	  
    }
    int src_node; cin >> src_node;
    vector<int> dist_from_src(N+1, INT_MAX);
    priority_queue< int,
		    vector< pair<int, int> >,
		    DistLEQ > next_node_queue;    
    
    next_node_queue.push(make_pair(src_node, 0));
    while ( ! next_node_queue.empty() ) {
      auto node_and_weight = next_node_queue.top();
      next_node_queue.pop();      
      int node = node_and_weight.first;
      int dist = node_and_weight.second;
      if ( dist > dist_from_src[node] )
	continue; // pruning
      dist_from_src[node] = dist;
      
      // Update the weights of its neighbors, if smaller
      for (auto it = weighted_edge_list[node].begin();
	   it != weighted_edge_list[node].end(); ++it) {
	int next_node = it->first;
	int next_dist = dist + it->second;
	if ( next_dist < dist_from_src[next_node] ) {
	  dist_from_src[next_node] = next_dist; // pruning 
	  next_node_queue.push(make_pair(next_node, next_dist));
	}
      }
    }

    for (int u = 1; u <= N; ++u) {
      if ( u != src_node ) {
	int dist = dist_from_src[u];
	if ( INT_MAX == dist )
	  cout << -1 << " ";
	else
	  cout << dist << " ";
      }
    }
    cout << endl;
  }
  
  return 0;
}




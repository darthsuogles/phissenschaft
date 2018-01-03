#include <iostream>
#include <unordered_map>
#include <vector>
#include <queue>

using namespace std;

struct Node {
  char data;
  vector<Node*> nghbrs;

  Node(const char data, vector<Node*> &nghbrs)
    : data(data), nghbrs(nghbrs) {}
  Node(const char data): data(data) {}
};

struct Graph {
  unordered_map<char, Node*> adjListMap;

  void addNeighbors(char tgtLabel, vector<char> nodeLabels) {
    vector<Node*> nghbrs;
    for (auto it = nodeLabels.begin(); it != nodeLabels.end(); ++it) {
      nghbrs.push_back(adjListMap[*it]);
    }
    Node *tgt = adjListMap[tgtLabel];
    tgt->nghbrs.insert(tgt->nghbrs.end(), nghbrs.begin(), nghbrs.end());
  }
};

void dfsAux(Graph &G, Node *node,
	    unordered_map<char, char> &visited_from) {
  // if ( visited_from.count(node->data) > 0 )
  //   return;
  cout << node->data << endl;
  for (auto it = node->nghbrs.begin();
       it != node->nghbrs.end(); ++it) {
    Node *next = *it;
    if ( visited_from.count(next->data) > 0 )
      continue;
    visited_from[next->data] = node->data;
    dfsAux(G, next, visited_from);
  }
}

void dfs(Graph &G, const char nodeLabel) {
  int N = G.adjListMap.size();
  unordered_map<char, char> visited_from;
  visited_from[nodeLabel] = '$';
  dfsAux(G, G.adjListMap[nodeLabel], visited_from);
}

void bfs(Graph &G, const char nodeLabel) {  
  unordered_map<char, char> visited_from;
  visited_from[nodeLabel] = '$';
  queue<Node *> node_queue;
  node_queue.push(G.adjListMap[nodeLabel]);
  while ( ! node_queue.empty() ) {
    Node *curr = node_queue.front();
    node_queue.pop();
    cout << curr->data << " ";
    for (auto it = curr->nghbrs.begin();
	 it != curr->nghbrs.end(); ++it) {
      Node *next = *it;
      if ( visited_from.count(next->data) > 0 )
	continue;
      visited_from[next->data] = curr->data;
      node_queue.push(next);
    }
  }
}

void ccompAux(Graph &G, Node *node,
	      const unsigned int cidx,
	      unordered_map<char, unsigned int> &ccLabel) {
  if ( NULL == node )
    return;
  for (auto it = node->nghbrs.begin();
       it != node->nghbrs.end(); ++it) {
    Node *next = *it;
    if ( ccLabel.count(next->data) )
      continue;
    ccLabel[next->data] = cidx;
    ccompAux(G, next, cidx, ccLabel);
  }
}

void ccomp(Graph &G) {
  unordered_map<char, unsigned int> ccLabel;
  int cidx = 0;
  for (auto it = G.adjListMap.begin();
       it != G.adjListMap.end(); ++it) {
    char label = it->first;
    Node *node = it->second;    
    if ( ccLabel.count(label) > 0 )
      continue;
    ccompAux(G, node, cidx++, ccLabel);
  }
  for (auto it = ccLabel.begin();
       it != ccLabel.end(); ++it) {
    cout << it->first << " " << it->second << endl;
  }
}

int main() {
  
  Graph G;
  for (char ch = 'A'; ch <= 'H'; ++ch) {
    G.adjListMap[ch] = new Node(ch);
  }
  // G.addNeighbors('A', {'F', 'B'});
  // G.addNeighbors('B', {'A'});
  // G.addNeighbors('C', {'G', 'H', 'D', 'F'});
  // G.addNeighbors('D', {'C'});
  // G.addNeighbors('E', {'F'});
  // G.addNeighbors('F', {'E', 'A', 'G', 'C'});
  // G.addNeighbors('G', {'C', 'H', 'F'});
  // G.addNeighbors('H', {'C', 'G'});

  G.addNeighbors('A', {'B'});
  G.addNeighbors('B', {'F', 'A', 'E'});
  G.addNeighbors('C', {'G', 'D', 'F'});
  G.addNeighbors('D', {'H', 'C', 'G'});
  G.addNeighbors('E', {'B'});
  G.addNeighbors('F', {'B', 'G', 'C'});
  G.addNeighbors('G', {'C', 'F', 'H', 'D'});
  G.addNeighbors('H', {'D', 'G'});

  //dfs(G, 'A');
  //bfs(G, 'A');
  ccomp(G);
}

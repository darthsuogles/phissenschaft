#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;

/**
 * Definition for undirected graph.
 * struct UndirectedGraphNode {
 *     int label;
 *     vector<UndirectedGraphNode *> neighbors;
 *     UndirectedGraphNode(int x) : label(x) {};
 * };
 */

struct UndirectedGraphNode
{
  int label;
  vector<UndirectedGraphNode *> neighbors;
  UndirectedGraphNode(int x) : label(x) {};
};


class Solution
{
  UndirectedGraphNode* clone_graph(UndirectedGraphNode* node,
				   unordered_map<int, UndirectedGraphNode*> &node_map)
  {
    if ( NULL == node ) return NULL;
    int nid = node->label;
    if ( node_map.count(nid) != 0 ) return node_map[nid];
        
    UndirectedGraphNode *curr = new UndirectedGraphNode(nid);
    node_map[nid] = curr;
    for (auto it = node->neighbors.begin(); it != node->neighbors.end(); ++it)
      {
	curr->neighbors.push_back( clone_graph(*it, node_map) );
      }
    return curr;
  }
public:
  UndirectedGraphNode *cloneGraph(UndirectedGraphNode *node)
  {
    unordered_map<int, UndirectedGraphNode*> node_map;
    return clone_graph(node, node_map);
  }
};

int main()
{
  Solution sol;
  
}

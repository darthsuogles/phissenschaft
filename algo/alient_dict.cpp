#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <unordered_map>
#include <unordered_set>

using namespace std;

struct Node {
  int indeg;
  unordered_set<char> children;

  Node(): indeg(0) {}
};

class DiGraph {
private:
  unordered_map<char, Node*> adjListMap;

  Node *getNode(char ch) {
    Node *node;
    if ( 0 == adjListMap.count(ch) )
      node = adjListMap[ch] = new Node();
    else
      node = adjListMap[ch];
    return node;
  }

public:  
  void addEdge(char ch1, char ch2) {        
    getNode(ch1)->children.insert(ch2);
    ++getNode(ch2)->indeg;
  }

  void addNode(char ch) {
    getNode(ch);
  }

  string topoSearch() {
    string res;
    unordered_set<char> visited;
    for (auto it = adjListMap.begin(); it != adjListMap.end(); ++it) {
      char ch = it->first;
      Node *node = it->second;      
      if ( 0 == node->indeg ) {
	if ( visited.count(ch) > 0 )
	  continue;
	string val = tps(ch);
	if ( "" == val ) return "";
	visited.insert(val.begin(), val.end());
	res += val;
      }
    }
    return res;
  }

private:
  string tps(char curr) {
    Node *node = adjListMap[curr];
    //if ( NULL == node ) return "";
    assert(node != NULL);
    if ( node->children.empty() ) return string(1, curr);

    string res;
    for (auto it = node->children.begin(); it != node->children.end(); ++it) {
      // Depth-first search over all children nodes with zero in-degree
      char ch = *it;      
      if ( 0 == --adjListMap[ch]->indeg ) {
	string val = tps(ch);
	if ( val != "" )
	  res += val;
      }
    }
    if ( res != "" )
      res = string(1, curr) + res;
    return res;
  }
};

string decodeAlphabet(const vector<string> &words) {
  string res;
  if ( words.empty() ) return res;
  int len = words.size();
  if ( 1 == len ) return words[0];

  DiGraph graph;
  
  for (int i = 0; i + 1 < len; ++i) {
    string curr = words[i];
    string next = words[i+1];
    int n = min(curr.size(), next.size());
    int k = 0;
    for (; k < n; ++k) {
      int ch1 = curr[k], ch2 = next[k];
      graph.addNode(ch1);
      graph.addNode(ch2);
      if ( ch1 != ch2 ) {
	graph.addEdge(ch1, ch2);
	break;
      }
    }
    for (int j = k; j < curr.size(); ++j)
      graph.addNode(curr[j]);
    for (int j = k; j < next.size(); ++j)
      graph.addNode(next[j]);
  }
  
  return graph.topoSearch();
}

int main() {
  
  vector<string> words = {
    "wrt",
    "wrf",
    "er",
    "ett",
    "rftt",
    "dds",
  };
  cout << decodeAlphabet(words) << endl;
  cout << "-----------" << endl;
}

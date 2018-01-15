#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;

struct Node {
  int data;
  vector<Node*> children;

  Node(): data(-1) {}
  Node(int data): data(data) {}
  Node(int data, const vector<Node*> &children) :
    data(data), children(children) {}
};

void delete_tree(Node* root) {
  if ( NULL == root ) return;
  for (auto nt = root->children.begin();
       nt != root->children.end(); ++nt) {
    delete_tree(*nt);
    *nt = NULL;
  }
}

void print_tree(Node *root, int pref_len = 0) {
  if ( NULL == root ) return;
  for (int i = 0; i < pref_len; ++i)
    cout << " ";
  cout << " | " << root->data << endl;
  for (auto nt = root->children.begin();
       nt != root->children.end(); ++nt)
    print_tree(*nt, pref_len + 1);
}

int max_wt_indep_set(Node *root, bool with_root = true, bool init_cache = false) {
  if ( NULL == root ) return 0;

  static unordered_map<Node*, int> cache_with_root;
  static unordered_map<Node*, int> cache_sans_root;
  if ( init_cache ) {
    cache_with_root.clear();
    cache_sans_root.clear();
  }

  int max_val = 0;
  if ( with_root ) {
    max_val = root->data;
    int max_val_grand = 0;
    for (auto nt = root->children.begin();
	 nt != root->children.end(); ++nt) {
      Node *curr = *nt;
      if ( cache_sans_root.count(curr) > 0 )
	max_val += cache_sans_root[curr];
      else
	max_val += (cache_sans_root[curr] = max_wt_indep_set(*nt, false));
    }
    max_val += max_val_grand;
  }
  int max_val_children = 0;
  for (auto nt = root->children.begin();
       nt != root->children.end(); ++nt) {
    Node *curr = *nt;
    if ( cache_with_root.count(curr) > 0 )
      max_val_children += cache_with_root[curr];
    else
      max_val_children += (cache_with_root[curr] = max_wt_indep_set(*nt, true));
  }
  max_val = max(max_val, max_val_children);
  return max_val;
}

int main() {
  Node *root = new Node(10, {
      new Node(20, {
	  new Node(40),
	  new Node(50, {
		new Node(70),
		new Node(80)})
	    }),
      new Node(30, {new Node(60)})
	});

  print_tree(root);
  cout << max_wt_indep_set(root, true, true) << endl;
  delete_tree(root);
}

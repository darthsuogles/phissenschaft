/**
 * The original problem assumes a binary search tree
 * This implementation provides one for general trees
 */
#include <iostream>
#include "TreeNode.hpp"

using namespace std;

bool find(TreeNode *root, TreeNode *node) {
	if (NULL == root) return false;
	if (root == node) return true;
	return find(root->left, node) || find(root->right, node);
}

// Reverse path to the root
bool find_path(TreeNode *root, TreeNode *node, vector<TreeNode*> &trail) {
	if (NULL == root) return false;
	if (root == node || 
		find_path(root->left, node, trail) ||
		find_path(root->right, node, trail)) {
		trail.push_back(root); return true;
	}
	return false;
}

TreeNode* lowestCommonAncestor(TreeNode *root, TreeNode *p, TreeNode *q) {
	vector<TreeNode*> trail_p; find_path(root, p, trail_p);
	if (trail_p.empty()) return NULL;
	vector<TreeNode*> trail_q; find_path(root, q, trail_q);
	if (trail_q.empty()) return NULL;
	TreeNode *anc = root;
	for (auto it_p = trail_p.rbegin(), it_q = trail_q.rbegin();
		 it_p != trail_p.rend() && it_q != trail_q.rend();
		 ++it_p, ++it_q) {
		if (*it_p != *it_q) break;
		anc = *it_p;
	}
	return anc;
}

TreeNode* lowestCommonAncestorBF(TreeNode *root, TreeNode *p, TreeNode *q) {
	if (NULL == root) return NULL;
	if (root == p) {
		if (find(root, q)) return root;
		return NULL;
	} 
	if (root == q) {
		if (find(root, p)) return root;
		return NULL;
	}
	
	bool lp = find(root->left, p), lq = find(root->left, q);
	if (lp && lq) return lowestCommonAncestor(root->left, p, q);
	bool rp = find(root->right, p);
	if (lq && rp) return root;
	bool rq = find(root->right, q);
	if (lp && rq) return root;
	if (rp && rq) return lowestCommonAncestor(root->right, p, q);
	return NULL;
}


#define X INT_MIN

int main() {
	auto root = TreeNode::from({1, X, 2, 3});
	root->print();
	auto lca = lowestCommonAncestor(root, root->right, root->right->left);
	cout << lca->val << endl;
}

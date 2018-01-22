/**
 * Find bottom left value in a binary tree
 */

#include <vector>
#include <iostream>
#include "TreeNode.hpp"

using namespace std;

#define X INT_MIN

struct NodePos {
	TreeNode *node;
	int skew;
	int depth;

	NodePos(TreeNode *node, int skew, int depth)
		: node(node), skew(skew), depth(depth) {}
};

NodePos* search(NodePos *node_pos) {
	if (NULL == node_pos || NULL == node_pos->node) return NULL;
	auto node = node_pos->node;
	// Base case for leaf node
	if (NULL == node->left && NULL == node->right)
		return node_pos;

	// Find best amongst children nodes
	int skew = node_pos->skew;
	int depth = node_pos->depth;
	auto pos_left = search(new NodePos(node->left, skew - 1, depth + 1));
	auto pos_right = search(new NodePos(node->right, skew + 1, depth + 1));
	if (NULL == pos_left) return pos_left;
	if (NULL == pos_right) return pos_left;	

	if (pos_left->depth == pos_right->depth) {
		return (pos_left->skew <= pos_right->skew) ? 
			pos_left : pos_right;
	} else {
		return (pos_left->depth > pos_right->depth) ?
			pos_left : pos_right;
	}
}


int findBottomLeftValue(TreeNode *root) {
	auto node_pos = search(new NodePos(root, 0, 0));
	return node_pos->node->val;
}

int main() {
	auto root = TreeNode::from({1, 2, 3, 4, X, 5, 6, X, X, 7});
	root->print();
	int res = findBottomLeftValue(root);
	cout << res << endl;
}

#include <iostream>
#include <utility>
#include <vector>
#include "TreeNode.hpp"

using namespace std;

pair<TreeNode*, TreeNode*> lca_kern(TreeNode *root, TreeNode *p, TreeNode *q) {
    if (nullptr == root) return make_pair(nullptr, nullptr);
    // Candidates for p and q's ancestors
    TreeNode *anc_p = nullptr, *anc_q = nullptr;

    // First check if root matches any
    if (root == p) anc_p = root;
    if (root == q) anc_q = root;
    // Root matches both
    if (anc_p && anc_q) return make_pair(anc_p, anc_q);

    // The algorithm naturally extends to cases where
    // the tree has larger branch factor or we have more target nodes.
    for (auto node: {root->left, root->right}) {
        auto candidate = lca_kern(node, p, q);
        if (!anc_p) anc_p = get<0>(candidate);
        if (!anc_q) anc_q = get<1>(candidate);
        if (anc_p && anc_q) {
            if (anc_p == anc_q)  // lca is some subtree node
                return make_pair(anc_p, anc_q);
            // Otherwise, root is their common ancestor
            return make_pair(root, root);
        }
    }
    return make_pair(anc_p, anc_q);
}

TreeNode* lowestCommonAncestorCircuitBreaker(TreeNode *root, TreeNode *p, TreeNode *q) {
    if (nullptr == root) return nullptr;  // no match
    if (nullptr == p) return q;
    if (nullptr == q) return p;
    auto res = lca_kern(root, p, q);
    auto anc_p = get<0>(res);
    auto anc_q = get<1>(res);
    if (anc_p && anc_q) return anc_p;
    return nullptr; // no match
}

// If we assume that both p and q are present in the tree
TreeNode* lowestCommonAncestor(TreeNode *root, TreeNode *p, TreeNode *q) {
    // If q is in a subtree of p, then returning root here is safe.
    // If q is not in a subtree of p, then further search will return nothing.
    // Thus it is always safe to return the matching node.
    if (nullptr == root || p == root || q == root) return root;
    auto anc_l = lowestCommonAncestor(root->left, p, q);
    auto anc_r = lowestCommonAncestor(root->right, p, q);
    // Return root if both are non-empry
    return !anc_l ? anc_r : !anc_r ? anc_l : root;
}

#define X INT_MIN

int main() {
	auto root = TreeNode::from({1, X, 2, 3});
	root->print();
	auto lca = lowestCommonAncestor(root, root->right, root->right->left);
	cout << lca->val << endl;
}

/**
 * Check if tree is symmetric
 */
#include <iostream>
#include "TreeNode.hpp"
#include <vector>
#include <deque>

using namespace std;

#define X INT_MIN

bool isMirror(TreeNode *left, TreeNode *right) {
    if (NULL == left) return NULL == right;
    if (NULL == right) return false;
    if (left->val != right->val) 
        return false;
    if (isMirror(left->right, right->left))
        return isMirror(left->left, right->right);
    return false;
}

bool isSymmetricRec(TreeNode *root) {
    if (NULL == root) return true;
    return isMirror(root->left, root->right);
}

// Iteratively search each level of the tree traversal with BFS
bool isSymmetric(TreeNode *root) {
    if (NULL == root) return true;
    if (NULL == root->left) return NULL == root->right;
    if (NULL == root->right) return false;
    
    deque<TreeNode *> q0, q1;
    q0.push_back(root->left);
    q1.push_back(root->right);
    while ((! q0.empty()) && (! q1.empty())) {
        int n = q0.size();        
        if (n != q1.size()) return false;
        for (int i = 0, j = n - 1; i < n; ++i, --j) {
            auto u = q0[i], v = q1[j];            
            if (NULL == u) {
                if (NULL == v) continue;
                return false;
            }
            if (NULL == v) return false;
            if (u->val != v->val)
                return false;
        }
        for (; n > 0; --n) {
            auto node0 = q0.front(); q0.pop_front();
            auto node1 = q1.front(); q1.pop_front();
            if (NULL != node0) {
                q0.push_back(node0->left);
                q0.push_back(node0->right);
            }
            if (NULL != node1) {
                q1.push_back(node1->left);
                q1.push_back(node1->right);
            }
        }
    }
    return q0.empty() && q1.empty();
}

void TEST(vector<int> arr, bool tgt) {
    auto root = TreeNode::from(arr);
    auto res = isSymmetric(root);
    if (res != tgt)
        cout << "ERROR: " << res << " but expect " << tgt << endl;
    else
        cout << "Ok" << endl;
}

int main() {
    TEST({1,2,2,3,4,4,3}, true);
    TEST({1, 2, 2, X, 3, X, 3}, false);
    TEST({2,3,3,4,5,5,4,X,X,8,9,9,8}, true);
}

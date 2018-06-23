#include <iostream>
#include <utility>
#include <vector>
#include <stack>
#include <unordered_set>
#include "TreeNode.hpp"

using namespace std;

// https://leetcode.com/problems/kth-smallest-element-in-a-bst/discuss/
class Solution {
    // Not thread-safe
    TreeNode *target_;

    int find_kth_dfs(TreeNode *root, int &k) {
        if (nullptr == root) return -1;
        int val = find_kth_dfs(root->left, k);
        return !k ? val : !--k ? root->val : find_kth_dfs(root->right, k);
    }

    // Return size of the subtree if not found, otherwise return -1
    int find_kth_dfs_const(TreeNode *root, const int k) {
        if (nullptr == root) return 0;
        int maybe_left_size = find_kth_dfs_const(root->left, k);
        if (-1 == maybe_left_size) return -1;
        if (maybe_left_size + 1 == k) {
            target_ = root; return -1;
        }
        int maybe_right_size =
            find_kth_dfs_const(root->right, k - maybe_left_size - 1);
        if (-1 == maybe_right_size) return -1;
        // If we have not found the target, return the subtree's size
        return maybe_left_size + 1 + maybe_right_size;
    }
public:
    int kthSmallestConst(TreeNode *root, int k) {
        if (nullptr == root) return -1;
        target_ = nullptr;
        find_kth_dfs_const(root, k);
        if (nullptr == target_) return -1;
        return target_->val;
    }

    int kthSmallest(TreeNode *root, int k) {
        if (nullptr == root) return -1;
        return find_kth_dfs(root, k);
    }

    int kthSmallestWithDFI(TreeNode *root, int k) {
        if (nullptr == root) return -1;
        stack<TreeNode *> dfs_stack;
        unordered_set<TreeNode *> visited;
        visited.insert(nullptr);
        int idx = 0;
        dfs_stack.push(root);
        while (!dfs_stack.empty()) {
            auto node = dfs_stack.top();
            // Just make sure this node is never pushed again
            visited.insert(node);
            if (!visited.count(node->left)) {
                dfs_stack.push(node->left);
                continue;
            }
            dfs_stack.pop();
            if (++idx == k) return node->val;
            if (!visited.count(node->right)) {
                dfs_stack.push(node->right);
            }
        }
        return -1;
    }

    // Non-recursive version: explicitly using stack, no extra storage
    int kthSmallestNonStockage(TreeNode *root, int k) {
        stack<TreeNode *> dfs_stack;
        auto node = root;
        while (node || !dfs_stack.empty()) {
            while (node) {
                dfs_stack.push(node);
                node = node->left;
            }
            node = dfs_stack.top();
            if (!--k) return node->val;
            dfs_stack.pop();
            node = node->right;
        }
        return -1;
    }
};

Solution sol;

void TEST(vector<int> nums, int k, int tgt) {
    auto root = TreeNode::from(nums);
    root->print();
    int res = sol.kthSmallest(root, k);
    if (res != tgt) {
        cout << "FAIL: got " << res << " != (ref) " << tgt << endl;
    } else {
        cout << "PASS" << endl;
    }
}

int main() {
    TEST({2, 1, 3}, 1, 1);
    TEST({1}, 1, 1);
    TEST({4, 3, 7, 1, 2, 5}, 1, 1);
    TEST({4, 2, 7, 1, 3, 5}, 4, 4);
}

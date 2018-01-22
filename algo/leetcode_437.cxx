#include <iostream>
#include <vector>
#include "TreeNode.hpp" 

using namespace std;

int psum(TreeNode*, int, int);

// Only continue the sum path search
int psum_cont(TreeNode *root, int sum) {
    if (NULL == root) return 0;
    int res = 0;
    int next = sum - root->val;
    if (0 == next) ++res;
    return res + 
        psum_cont(root->left, next) + 
        psum_cont(root->right, next);
}

// Both continue and initiate new search
int psum(TreeNode *root, int sum_cont, int sum_orig) {
    if (NULL == root) return 0;

    int res = 0;
    int next_cont = sum_cont - root->val;
    int next_orig = sum_orig - root->val;
    if (0 == next_cont) ++res;
    if (0 == next_orig) ++res;

    res += 
        psum_cont(root->left, next_cont) + 
        psum_cont(root->right, next_cont) +
        psum(root->left, next_orig, sum_orig) + 
        psum(root->right, next_orig, sum_orig);
    return res;
}

// Only initiate new search
int pathSum(TreeNode *root, int sum) {
    if (NULL == root) return 0;
    int next = sum - root->val;
    return (0 == next ? 1 : 0) + 
        psum(root->left, next, sum) + 
        psum(root->right, next, sum);
}

void TEST(vector<int> vals, int sum, int tgt) {
    auto root = TreeNode::from(vals);
    root->print();
    int res = pathSum(root, sum);
    if (tgt != res) 
        cout << "ERROR " << res << " != " << tgt << endl;
    else 
        cout << "OK" << endl;
    cout << "-----------------------------" << endl;    
    delete root;
}

#define null INT_MIN

int main() {
    TEST({10, 5, -3, 3, 2, null, 11, 3, -2, null, 1}, 8, 3);
    TEST({1, null, 2, null, 3, null, 4, null, 5}, 3, 2);
    TEST({1, null, 2, null, 3, null, 4, null, 5}, 7, 1);
}

/**
 * Path sum
 */
#include <iostream>
#include "TreeNode.hpp"
#include <vector>

using namespace std;

bool hasPathSum(TreeNode *root, int sum) {
    if (NULL == root) return false;
    int next_sum = sum - root->val;
    // Only check zeros when reaching a leaf node
    if ((NULL == root->left) && (NULL == root->right))
        return 0 == next_sum;
    return hasPathSum(root->left, next_sum) || hasPathSum(root->right, next_sum);
}

void TEST(vector<int> arr, int sum, bool expected) {
    auto root = TreeNode::from(arr);    
    root->print();
    if (expected != hasPathSum(root, sum)) {
        cout << "expected " << expected << " but got " << !expected << endl;
    } else
        cout << "Ok" << endl;
}

int main() {
#define X INT_MIN
    TEST({5,4,8,11,X,13,4,7,2,X,X,X,1}, 22, true);
    TEST({}, 0, false);
    TEST({1,-2,-3,1,3,-2,X,-1}, -1, true);
}

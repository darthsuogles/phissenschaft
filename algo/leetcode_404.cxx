#include <iostream>
#include <vector>
#include <queue>
#include "TreeNode.hpp"

using namespace std;

int sumOfLeftLeaves(TreeNode *root) {   
    int res = 0;
    if (NULL == root) return 0;
    if (NULL == root->left) 
        return sumOfLeftLeaves(root->right);

    if (isLeaf(root->left))  // the base case
        res += root->left->val;
    else 
        res = sumOfLeftLeaves(root->left);

    res += sumOfLeftLeaves(root->right);    
    return res;
}

#define TEST(...) {                             \
        TreeNode *root = constr(vector<int> {__VA_ARGS__});   \
        print(root); \
        cout << "RESULT: " << sumOfLeftLeaves(root) << endl << endl;  \
        del(root); }

int main() {
    TEST(1, -1, 2);
    TEST(3, 9, 20, -1, -1, 15, 7);
    TEST(1, 2);
}

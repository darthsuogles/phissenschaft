#include <iostream>
#include <vector>
#include "TreeNode.hpp"

using namespace std;

vector<int> getNumbers(TreeNode *root, int prefix) {
    vector<int> res;
    if (NULL == root) return res;

    int a = root->val;
    while (a > 0) {
        prefix *= 10;
        a /= 10;
    }
    prefix += root->val;
    if (NULL == root->left && NULL == root->right) {
        res.push_back(prefix);
        return res;
    }
    res = getNumbers(root->left, prefix);
    auto res_aux = getNumbers(root->right, prefix);
    res.insert(res.end(), res_aux.begin(), res_aux.end());
    return res;
}

int pathSum(TreeNode *root) {
    int res = 0;
    auto numbers = getNumbers(root, 0);
    for (auto a: numbers) {
        cout << a << endl;
        res += a;
    }
    return res;
}

void TEST(vector<int> arr) {
    auto root = TreeNode::from(arr);
    root->print();
    int psum = pathSum(root);
    cout << "path sum: " << psum << endl;
}

int main() {
    TEST({3, 2, 5});
    TEST({3, 5, 14});
}

#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include "TreeNode.hpp"

using namespace std;

typedef unordered_map< int, long > counter_t;

void get_subtree_counter(TreeNode *root, counter_t& cntr, long& root_cnt) {
    if (NULL == root) return;
    long sub_cnt = root->val;
    if (NULL != root->left)
        get_subtree_counter(root->left, cntr, sub_cnt);
    if (NULL != root->right)
        get_subtree_counter(root->right, cntr, sub_cnt);
    root_cnt += sub_cnt;
    ++cntr[sub_cnt];    
    return;
}

vector<int> findFrequentTreeSum(TreeNode *root) {
    vector<int> res;
    counter_t cntr;
    long root_cnt = 0;
    get_subtree_counter(root, cntr, root_cnt);    
    long max_cnts = -1;
    for (auto &p: cntr) {
        max_cnts = max(max_cnts, p.second);
    }
    for (auto &p: cntr) {
        if (p.second != max_cnts) continue;
        res.push_back(p.first);
    }
    return res;
}


void TEST(vector<int> arr) {
    auto root = TreeNode::from(arr);
    auto res = findFrequentTreeSum(root);
    for (auto a: res) cout << a << " ";
    cout << endl;
    cout << "------------------" << endl;
}

int main() {
    TEST({5, 2, -3});
    TEST({5, 2, -5});
}

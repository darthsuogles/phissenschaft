#include <iostream>
#include "TreeNode.hpp"
#include <deque>

using namespace std;

vector< vector<int> > zigzagLevelOrder(TreeNode *root) {    
    vector< vector<int> > level_repr;
    if (NULL == root) return level_repr;
    deque<TreeNode *> Q;
    Q.push_back(root);
    bool is_reverse = false;
    while (! Q.empty()) {
        vector<int> repr;
        if (is_reverse) {
            for (auto it = Q.rbegin(); it != Q.rend(); ++it) 
                repr.push_back((*it)->val);        
        }
        else {
            for (auto it = Q.begin(); it != Q.end(); ++it) 
                repr.push_back((*it)->val);
        }
        level_repr.push_back(repr);
        is_reverse = ! is_reverse;
        int n = Q.size();
        for (; n > 0; --n) {
            auto node = Q.front(); Q.pop_front();
            if (node->left)
                Q.push_back(node->left);
            if (node->right)
                Q.push_back(node->right);
        }
    }
    return level_repr;
}

void TEST(vector<int> arr) {
    auto root = TreeNode::from(arr);
    auto res = zigzagLevelOrder(root);
    for (auto &repr: res) {
        for (auto a: repr) cout << a << " ";
        cout << endl;
    }
}

int main() {
#define X INT_MIN
    TEST({3,9,20,X,X,15,7});
}

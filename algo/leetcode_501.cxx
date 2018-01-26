/**
 * Find mode 
 */

#include <iostream>
#include <vector>
#include <map>
#include "TreeNode.hpp"

using namespace std;

struct ValCnt {
    int val;
    int cnt;
    ValCnt(): val(INT_MIN), cnt(0) {}
    ValCnt(int v, int c): val(v), cnt(c) {}
};

struct MaxMinMode {
    ValCnt max;
    ValCnt min;
    vector<int> modes;
    int freq;
    MaxMinMode(): max(INT_MAX, 0), min(INT_MIN, 0), freq(0) {}
};

// This algorithm works for generic binary tree
MaxMinMode findMaxMinMode(TreeNode *root) {
    MaxMinMode res;
    if (NULL == root) return res;

    map<int, int> freq_tbl;
    int root_freq = 1;
    
    if (NULL != root->left) {
        auto res_left = findMaxMinMode(root->left);
        if (res_left.max.val == root->val)
            root_freq += res_left.max.cnt;
        res.min = res_left.min;
        for (auto a: res_left.modes) freq_tbl[a] += res_left.freq;
    } else
        res.min.val = root->val;

    if (NULL != root->right) {
        auto res_right = findMaxMinMode(root->right);
        if (res_right.min.val == root->val)
            root_freq += res_right.min.cnt;        
        res.max = res_right.max;
        for (auto a: res_right.modes) freq_tbl[a] += res_right.freq;
    } else
        res.max.val = root->val;

    freq_tbl[root->val] = root_freq;
    int max_cnt = -1;
    for (auto vcnt: freq_tbl) {
        max_cnt = max(vcnt.second, max_cnt);
    }
    if (max_cnt < root_freq) {
        res.modes.push_back(root->val);
        res.freq = root_freq;
    } else {
        res.freq = max_cnt;
        for (auto vcnt: freq_tbl) {
            if (max_cnt != vcnt.second) continue;
            res.modes.push_back(vcnt.first);
        }
    }
    
    if (res.min.val == root->val) res.min.cnt = root_freq;
    if (res.max.val == root->val) res.max.cnt = root_freq;

    // cout << "mode freq " << res.freq << ", "
    //      << "min " << res.min.val << " @ " << res.min.cnt << ", "
    //      << "max " << res.max.val << " @ " << res.max.cnt << ", "
    //      << endl << "\t";
    // for (auto a: res.modes) cout << a << " "; cout << endl;
    return res;
}

/**
 * This algorithm works for general binary tree structure
 */
vector<int> findModeGen(TreeNode *root) {
    auto res = findMaxMinMode(root);
    return res.modes;
}

/** 
 * This algorithm works for binary search tree
 */
class TreeNodeIter {
    TreeNode* first;
    TreeNode* last;
    TreeNode* curr;
public:
    TreeNodeIter(TreeNode *root): first(NULL), last(NULL), curr(NULL) {
        if (NULL == root) return;
        first = last = root;

        if (NULL != root->left) {
            auto iter_left = new TreeNodeIter(root->left);
            first = iter_left->first;
            iter_left->last->right = root;
            delete iter_left;
        } 

        if (NULL != root->right) {
            auto iter_right = new TreeNodeIter(root->right);
            last = iter_right->last;
            root->right = iter_right->first;
            delete iter_right;
        } 
        curr = first;
    }
    
    bool hasNext() { return NULL != curr; }
    TreeNode* getNext() { 
        auto res = curr;
        curr = curr->right;
        return res;
    }
    void reset() { curr = first; }
};

vector<int> findModeWithIter(TreeNode *root) {
    vector<int> res;
    if (NULL == root) return res;
    auto iter = new TreeNodeIter(root);
    int max_cnt = 0;

    int prev_cnt = 1;
    int prev = iter->getNext()->val;
    while (iter->hasNext()) {
        int curr = iter->getNext()->val;
        if (curr == prev) {
            max_cnt = max(max_cnt, ++prev_cnt);
        } else {            
            prev_cnt = 1;
            prev = curr;
        }            
    }
    
    iter->reset();
    prev_cnt = 1;
    prev = iter->getNext()->val;
    while (iter->hasNext()) {
        int curr = iter->getNext()->val;
        if (curr == prev) {
            if (++prev_cnt == max_cnt)
                res.push_back(curr);
        } else {
            prev_cnt = 1;
            prev = curr;
        }
    }
    return res;
}


void dfs(TreeNode *root, int& prev, int& prev_cnt, int& max_cnt) {
    if (NULL == root) return;
    
    dfs(root->left, prev, prev_cnt, max_cnt);
    if (prev == root->val) {
         ++prev_cnt;
    } else {
        prev = root->val;
        prev_cnt = 1;
    }
    max_cnt = max(max_cnt, prev_cnt);
    dfs(root->right, prev, prev_cnt, max_cnt);
}

void dfs_attach(TreeNode *root, int& prev, int& prev_cnt, const int max_cnt, vector<int>& res) {
    if (NULL == root) return;
    
    dfs_attach(root->left, prev, prev_cnt, max_cnt, res);
    if (prev == root->val) {
        ++prev_cnt;
    } else {
        prev = root->val;
        prev_cnt = 1;
    }
    if (max_cnt == prev_cnt) res.push_back(root->val);
    dfs_attach(root->right, prev, prev_cnt, max_cnt, res);
}

vector<int> findMode(TreeNode *root) {
    vector<int> res;
    if (NULL == root) return res;
    int prev = root->val + 1;
    int prev_cnt = 0;
    int max_cnt = 0;
    dfs(root, prev, prev_cnt, max_cnt);

    prev = root->val + 1;
    prev_cnt = 0;
    dfs_attach(root, prev, prev_cnt, max_cnt, res);
    return res;
}



#define X INT_MIN

void TEST(vector<int> tree_enc) {
    auto root = TreeNode::from(tree_enc);
    //root->print();
    for (auto a: findMode(root))
        cout << a << " ";
    cout << endl;
}

int main() {
    TEST({3, X, 2, 3});
    // TEST({6,9,8,0,4,7,9,X,X,2,6});
    // TEST({7});
    TEST({2147483647});
}

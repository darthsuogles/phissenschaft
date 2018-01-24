#include <iostream>
#include <vector>
#include <queue>

struct TreeNode;
TreeNode* constr(std::vector<int>);
void printNode(TreeNode*);
void del(TreeNode*);

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x): val(x), left(NULL), right(NULL) {}
    ~TreeNode() { delete left; delete right; left = right = NULL; }
    static TreeNode* from(std::vector<int> vals) { return constr(vals); }
    void print() { printNode(this); }
};

bool isLeaf(TreeNode *root) {
    if (NULL == root) return false;
    if (NULL == root->left) return NULL == root->right;
    return false;
}

TreeNode* constr(std::vector<int> vals) {
    if (vals.empty()) return NULL;
    TreeNode *root = new TreeNode(vals[0]);
    std::queue<TreeNode *> ss;
    ss.push(root);
    for (int i = 1; i < vals.size(); i += 2) {
        TreeNode *curr = ss.front(); ss.pop(); 
        if (INT_MIN != vals[i]) {
            curr->left = new TreeNode(vals[i]);
            ss.push(curr->left);
        }
        if (i + 1 == vals.size()) break;
        if (INT_MIN != vals[i+1]) {
            curr->right = new TreeNode(vals[i+1]);
            ss.push(curr->right);
        }
    }
    return root;
}

void printNode(TreeNode *root, int spc) {
    for (int i = 0; i < spc; ++i) std::cout << " ";
    if (NULL == root) {
        std::cout << "|--*" << std::endl;
        return;
    }
    std::cout << "|= " << root->val << std::endl;
    if (isLeaf(root)) return;
    printNode(root->left, spc + 2);
    printNode(root->right, spc + 2);
}

void printNode(TreeNode *root) {
    std::cout << "TREE" << std::endl;;
    printNode(root, 1);
}

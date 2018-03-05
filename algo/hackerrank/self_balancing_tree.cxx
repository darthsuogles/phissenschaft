#include <iostream>
#include "../TreeNode.hpp"

using namespace std;

struct AvlTreeNode: public TreeNode {
    int ht;
    AvlTreeNode(): TreeNode(INT_MAX), ht(0) {}
    AvlTreeNode(int val, int ht = 0): TreeNode(val), ht(ht) {}
};

using node = AvlTreeNode;

inline node* new_node(int val, int ht = 0) {
    auto curr = new node();
    curr->val = val; curr->ht = ht;
    return curr;
}

#define HT(root) ((nullptr != (root)) ? ((node *) (root))->ht : -1)

inline void adjust_ht(node *root) {
    if (nullptr == root) return;
    root->ht = 1 + max(HT(root->left), HT(root->right));
}

#define HEIGHT_BLOCK(root)                                              \
    int ht_left, ht_right, factor; {                                    \
        ht_left = HT((root)->left);                                     \
        ht_right = HT((root)->right);                                   \
        factor = ht_left - ht_right;                                    \
    }                                                                   \

#define INIT_BLOCK                                  \
    if (nullptr == root) return root;               \
    node *new_root = nullptr; HEIGHT_BLOCK(root)

#define FINI_BLOCK                                          \
    adjust_ht(root); adjust_ht(new_root); return new_root


node* adjust_left_right(node *root) {
    INIT_BLOCK;
    if (factor != -1 || nullptr == root->right) return root;
    new_root = (node *) root->right;
    root->right = new_root->left;
    new_root->left = root;
    FINI_BLOCK;
}

node *adjust_left_left(node *root) {
    INIT_BLOCK;
    if (factor != 2 || nullptr == root->left) return root;
    new_root = (node *) root->left;
    root->left = new_root->right;
    new_root->right = root;
    FINI_BLOCK;
}

node *adjust_right_left(node *root) {
    INIT_BLOCK;
    if (factor != 1 || nullptr == root->left) return root;
    new_root = (node *) root->left;
    root->left = new_root->right;
    new_root->right = root;
    FINI_BLOCK;
}

node *adjust_right_right(node *root) {
    INIT_BLOCK;
    if (factor != -2 || nullptr == root->right) return root;
    new_root = (node *) root->right;
    root->right = new_root->left;
    new_root->left = root;
    FINI_BLOCK;
}

node* balance(node *root) {
    if (nullptr == root) return root;
    HEIGHT_BLOCK(root);
    switch (factor) {
    case 2: // left-right? => left-left
        root->left = adjust_left_right((node *) root->left);
        root = adjust_left_left(root);
        break;
    case -2: // right-left? => right-right
        root->right = adjust_right_left((node *) root->right);
        root = adjust_right_right(root);
        break;
    }
    adjust_ht(root);
    return root;
}

node* insert(node *root, int val) {
    if (nullptr == root) return root;
    if (val <= root->val) {
        root->left = root->left ?
            insert((node *)root->left, val) :
            new_node(val, 0);
        adjust_ht((node *)root->left);
    } else {
        root->right = root->right ?
            insert((node *)root->right, val) :
            new_node(val, 0);
        adjust_ht((node *)root->right);
    }
    adjust_ht(root);
    return balance(root);
}

int main() {
    auto root = new AvlTreeNode(3);
    root = insert(root, 2);
    root = insert(root, 4);
    root = insert(root, 5);
    root = insert(root, 6);
    root->print();
}

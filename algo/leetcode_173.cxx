#include <iostream>
#include <stack>
#include <cassert>
#include "TreeNode.hpp"

using namespace std;

/**
 * Destroy the tree structure
 * Constant memory (after tree construction)
 * Need traverse the tree first
 */
class BSTFlatten {
    TreeNode* first;
    TreeNode* last;
    TreeNode* curr;
public:
    BSTFlatten(TreeNode *root): first(NULL), last(NULL), curr(NULL) {
        if (NULL == root) return;
        first = last = root;

        if (NULL != root->left) {
            auto iter_left = new BSTFlatten(root->left);
            first = iter_left->first;
            iter_left->last->right = root;
            delete iter_left;
        }

        if (NULL != root->right) {
            auto iter_right = new BSTFlatten(root->right);
            last = iter_right->last;
            root->right = iter_right->first;
            delete iter_right;
        }
        curr = first;
    }

    bool hasNext() { return NULL != curr; }
    int next() {
        auto res = curr;
        curr = curr->right;
        return res->val;
    }

};

/**
 * Original tree structure intact
 * Create the iterator lazily
 * Space proportional to depth (only one-sided expansion)
 */
class BSTFlattenLazy {
    BSTFlattenLazy* iter;
    TreeNode* root;
public:
    BSTFlattenLazy(TreeNode *root): root(root), iter(NULL) {}

    bool hasNext() {
        if (NULL == root) {
            if (NULL == iter) return false;
            return iter->hasNext();
        }
        return true;
    }

    int next() {
        int curr = INT_MAX;
        if (NULL == iter) {
            if (NULL != root->left) {
                iter = new BSTFlattenLazy(root->left);
                curr = iter->next();
            } else {
                curr = root->val;
                if (NULL != root->right)
                    iter = new BSTFlattenLazy(root->right);
                root = NULL;
            }
        } else {
            if (iter->hasNext()) {
                curr = iter->next();
            } else {
                delete iter; iter = NULL;
                if (NULL != root) {
                    curr = root->val;
                    if (NULL != root->right)
                        iter = new BSTFlattenLazy(root->right);
                    root = NULL;
                }
            }
        }
        return curr;
    }
};


/**
 * Keep the original structure
 * Space proportional to depth
 */
class BSTIterator {
private:
    // https://google.github.io/styleguide/cppguide.html#Variable_Names
    // Maintain the depth first search traversal info
    stack<TreeNode*> dfs_stack_;

    bool is_reverse_;

    void fill_stack_backward(TreeNode *root) {
        for (auto curr = root; curr; curr = curr->right)
            dfs_stack_.push(curr);
    }

    void fill_stack_forward(TreeNode *root) {
        for (auto curr = root; curr; curr = curr->left)
            dfs_stack_.push(curr);
    }

public:
    BSTIterator(TreeNode *root, bool is_reverse = false) {
        if (NULL == root) return;
        is_reverse_ = is_reverse;
        if (is_reverse)
            fill_stack_backward(root);
        else
            fill_stack_forward(root);
    }

    bool hasNext() {
        return ! dfs_stack_.empty();
    }

    int next() {
        auto res = dfs_stack_.top();
        dfs_stack_.pop();
        if (is_reverse_)
            fill_stack_backward(res->left);
        else
            fill_stack_forward(res->right);
        return res->val;
    }
};

class BSTParenIter {
    /**
     * No parent pointer, log(depth) time/space complexity
     */
private:
    stack<TreeNode *> parents_;
    TreeNode *last_;
    TreeNode *prev_;
    TreeNode *curr_;

public:
    BSTParenIter(TreeNode *root) {
        last_ = new TreeNode(-1);
        parents_.push(last_);
        prev_ = last_;
        last_->right = last_->left = nullptr;
        curr_ = root;
    }

    bool hasNext() {
        if (!curr_) return false;
        while (prev_ == curr_->right) {
            prev_ = curr_;
            curr_ = parents_.top();
            parents_.pop();
        }
        return curr_ != last_;
    }

    int next() {
        assert(hasNext());
        if (prev_ != curr_->left) {
            for (auto node = curr_->left; node; node = node->left) {
                parents_.push(prev_ = curr_);
                curr_ = node;
            }
        }
        int res = curr_->val;
        if (nullptr != curr_->right) {
            parents_.push(prev_ = curr_);
            curr_ = curr_->right;
        } else {
            prev_ = nullptr;
        }
        return res;
    }
};

#define X INT_MIN

void TEST(vector<int> tree_enc) {
    cout << "#------- TEST BEGIN -------" << endl;
    auto root = TreeNode::from(tree_enc);
    //root->print();
    //auto i = BSTFlattenLazy(root);
    for (int idx = 0; idx < 1; ++idx) {
        //auto i = BSTIterator(root, 0 == idx);
        auto i = BSTParenIter(root);
        for (int cnts = 0; i.hasNext() && cnts < 1 + tree_enc.size() ; ++cnts)
            cout << i.next() << " ";
        cout << endl;
    }
    root->print();
}

int main() {
    TEST({2,1,3});
    TEST({6,1,8,0,4,7,9,X,X,2,6});
    // TEST({7});
    TEST({2147483647});
}

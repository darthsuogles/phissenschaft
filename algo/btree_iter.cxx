#include <iostream>
#include <vector>
#include "TreeNode.hpp"

using namespace std;


class InOrdIter {
    TreeNode *head;
    TreeNode *last;
    TreeNode *bridge;
    TreeNode *curr;
public:
    InOrdIter(TreeNode *root)
        : head(NULL), last(NULL), bridge(NULL), curr(NULL) {
        if (NULL == root) return;
        if (NULL != root->left) {
            InOrdIter *iter = new InOrdIter(root->left);
            head = iter->head;
            iter->last->right = root;
            root->left = iter->last;
        } else {
            head = root;
        }
        if (NULL != root->right) {
            InOrdIter *iter = new InOrdIter(root->right);
            iter->head->left = root;
            root->right = iter->head;
            last = iter->last;
        } else {
            last = root;
        }        
        curr = head;
        bridge = root;
    }

    bool has_next() {
        return (NULL != curr);
    }

    TreeNode* next() {
        if (! has_next()) return NULL;
        auto next = curr;
        curr = curr->right;
        return next;
    }
};


class PreOrdIter {
	int idx;
	TreeNode *curr;   
	TreeNode *last;
	TreeNode *bridge;

public:
	PreOrdIter(TreeNode *root)
		: idx(0), curr(root), last(NULL), bridge(NULL) {		
		if (NULL == root) {
			curr = last = NULL;
		} else {
			PreOrdIter *iter_left = NULL, *iter_right = NULL;
			if (NULL != root->left) {
				iter_left = new PreOrdIter(root->left);
				bridge = iter_left->last;
				bridge->right = root->right;
			}
			if (NULL != root->right)
				iter_right = new PreOrdIter(root->right);
			if (NULL != iter_right) 
				last = iter_right->last;
			else if (NULL != iter_left)
				last = iter_left->last;
			else
				last = root;
		}
	}

	~PreOrdIter() {
		if (NULL != bridge) {
			cout << "cleaning bridge: " 
				 << bridge->val;
			if (NULL != bridge->right)
				cout << " -> " << bridge->right->val;
			cout << endl;
			bridge->right = NULL;
		}
	}

	TreeNode* next() {
		TreeNode *node = curr;
		if (curr != last) {
			if (NULL != curr->left) 
				curr = curr->left;
			else 
				curr = curr->right;
		} else 
			curr = NULL;
		return node;
	}

	bool has_next() {
		if (NULL == curr) return false;
		return true;
	}
};

#define X INT_MIN

int main() {
	auto root = TreeNode::from({1, 4, 2, X, X, X, 3});
	root->print();
	//auto iter = new PreOrdIter(root);
    auto iter = new InOrdIter(root);
	while (iter->has_next()) {
		cout << iter->next()->val << " ";
	}
	// cout << endl;
	// delete iter;
	// root->print();
	//delete root;
}


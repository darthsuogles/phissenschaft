#include <iostream>
#include <map>
#include <vector>
#include <forward_list>
#include <cassert>
#include "TreeNode.hpp"

using namespace std;

template <typename K, typename V>
using kv_tbl_t = map< K, vector<V> >;

typedef kv_tbl_t<int, int> tbl_t;
//typedef list< forward_list<int> > vert_list_t;

void vert_append(TreeNode *root, int idx, tbl_t &tbl) {
	if (NULL == root) return;
	tbl[idx].push_back(root->val);
	vert_append(root->left, idx-1, tbl);
	vert_append(root->right, idx+1, tbl);
}

struct VertListNode;
void rm_post(VertListNode*);
void rm_prev(VertListNode*);

struct VertListNode {
    VertListNode *prev;
    VertListNode *post;
    vector<int> data;

    VertListNode(): prev(nullptr), post(nullptr) {}
    void del() {
        rm_prev(prev); prev = nullptr;
        rm_post(post); post = nullptr;
    }
    VertListNode* get_prev() {        
        if (nullptr == prev) {
            prev = new VertListNode();
            prev->post = this;
        }
        return prev;
    }    
    VertListNode* get_post() {
        if (nullptr == post) {
            post = new VertListNode();
            post->prev = this;
        }
        return post;
    }
    void append(int a) { data.push_back(a); }
};

void rm_post(VertListNode *node) {
    if (!node) return;
    rm_post(node->post);
    delete node;
}

void rm_prev(VertListNode *node) {
    if (!node) return;
    rm_prev(node->prev);
    delete node;
}


void vert_append(TreeNode *root, VertListNode *vert_list_node) {
    if (NULL == root) return;
    assert(vert_list_node != nullptr);
    vert_list_node->append(root->val);
    if (root->left)
        vert_append(root->left, vert_list_node->get_prev());
    if (root->right)
        vert_append(root->right, vert_list_node->get_post());
}

void vert_print(TreeNode *root) {
    auto node = new VertListNode();
    vert_append(root, node);
    while (node->prev != nullptr)
        node = node->prev;
    auto center = node;
    while (node != nullptr) {
        for (auto a: node->data) cout << a << " ";
        cout << endl;
        node = node->post;
    }
    center->del();
}

void vert_print_with_tbl(TreeNode *root) {
	tbl_t tbl;
	vert_append(root, 0, tbl);
	for (auto &pv: tbl) {
		for (auto a: pv.second) cout << a << " ";
		cout << endl;
	}
}

#define X INT_MIN

int main() {
	auto root = TreeNode::from({1,2,3,4,5,6,7,X,X,X,X,X,8,X,9});
	vert_print(root);
}

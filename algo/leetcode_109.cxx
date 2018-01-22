#include <iostream>
#include <vector>
#include "ListNode.hpp"
#include "TreeNode.hpp"

using namespace std;

typedef LnkLstNode<int> ListNode;

TreeNode* toBST(ListNode *head, int n) {
	if (NULL == head || 0 == n) return NULL;
	switch (n) {
	case 1:
		return new TreeNode(head->val);
	case 2:
		auto root = new TreeNode(head->val);
		root->right = new TreeNode(head->next->val);
		return root;		
	}
	int k = n / 2;
	auto node = head; // number of increments, thus not <= k
	for (int i = 0; i < k; ++i, node = node->next);
	auto root = new TreeNode(node->val);
	root->left = toBST(head, k);
	root->right = toBST(node->next, n-k-1);
	return root;
}

TreeNode* sortedListToBST(ListNode *head) {
	if (NULL == head) return NULL;
	int n = 0;
	auto node = head;
	for (; node != NULL; node = node->next, ++n);
	return toBST(head, n);
}

int main() {
	auto head = new ListNode({1,1,2,3,5,8,13,21});
	auto root = sortedListToBST(head);
	root->print();
}

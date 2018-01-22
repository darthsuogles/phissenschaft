/**
 * Grouping odd / even nodes in a linked list
 */

#include "ListNode.hpp"
#include <iostream>
#include <vector>

using namespace std;

typedef LnkLstNode<int> ListNode;

ListNode* oddEvenList(ListNode *head) {
	if (NULL == head) return NULL;
	if (NULL == head->next) return head;
	ListNode *odd_head = head;
	ListNode *even_head = head->next;
	ListNode *node_iter = head->next->next;
	ListNode *conn = head->next;
	for (bool is_even = false; 
		 node_iter != NULL; 
		 is_even = !is_even, node_iter = node_iter->next) {
		ListNode *&curr = is_even ? even_head : odd_head;
		curr = curr->next = node_iter;
	}
	odd_head->next = conn;
	even_head->next = NULL;
	return head;
}

int main() {
	auto head = new ListNode({1,2,3,4,5});
	head->print();
	head = oddEvenList(head);
	head->print();
	delete head;
}

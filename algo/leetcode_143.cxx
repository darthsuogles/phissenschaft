/**
 * Reorder linked list
 */

#include "ListNode.hpp"
#include <iostream>
#include <vector>
#include <stack>

using namespace std;

typedef LnkLstNode<int> ListNode;

void reorderListRec(ListNode *head) {
    if (nullptr == head) return;
    if (nullptr == head->next) return;
    auto curr = head;
    while (curr->next != nullptr) 
        curr = curr->next;
    auto last = curr;
    for (curr = head; curr->next != last; curr = curr->next);
    curr->next = nullptr;
    auto next = head->next;
    head->next = last;
    reorderListRec(next);
    last->next = next;
}

void reorderList(ListNode *head) {
    if (nullptr == head) return;
    if (nullptr == head->next) return;
    stack<ListNode *> node_stack;
    int len = 0;
    auto curr = head;
    for (; curr != nullptr; curr = curr->next, ++len);

    // Stack contains half (even) or 1 + half (odd)
    curr = head;
    for (int i = 0; i < len / 2; ++i) curr = curr->next;
    for (; curr != nullptr; curr = curr->next) 
        node_stack.push(curr);
    
    ListNode *front = head, *last = nullptr;
    while (! node_stack.empty()) {
        last = node_stack.top(); 
        node_stack.pop();
        curr = front; front = front->next;
        curr->next = last;
        last->next = front;        
    }    
    // Last from stack is always the last element
    last->next = nullptr;
}

void TEST(vector<int> elems) {
    auto head = new ListNode(elems);
    head->print();
    reorderList(head);
    head->print();
}

int main() {
    TEST({1,2,3,4});
    TEST({1,2,3,4,5});
}

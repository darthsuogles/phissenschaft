/**
 * Sorting linked list in O(n\log n) and constant space
 */
#include <iostream>
#include <vector>
#include <cassert>
#include "ListNode.hpp"

using namespace std;

typedef LnkLstNode<int> ListNode;

ListNode* sortList(ListNode *head) {
    if (NULL == head) return NULL;
    if (NULL == head->next) return head;

    int n = 0;
    auto curr = head;
    for (; curr != NULL; ++n) curr = curr->next;

    auto ghost = new ListNode(-1); ghost->next = head;
    curr = ghost;    
    for (int i = 0; i < n/2; ++i) curr = curr->next;
    auto half = curr->next;
    assert(NULL != half);

    curr->next = NULL;
    auto head_lo = sortList(head);
    auto head_hi = sortList(half);

    curr = ghost;
    while (head_lo != NULL && head_hi != NULL) {
        if (head_lo->val < head_hi->val) {
            curr->next = head_lo;
            head_lo = head_lo->next;
        } else {
            curr->next = head_hi;
            head_hi = head_hi->next;
        }
        curr = curr->next;
    }
    for (; head_lo != NULL; head_lo = head_lo->next) {
        curr->next = head_lo; curr = curr->next;
    }
    for (; head_hi != NULL; head_hi = head_hi->next) {
        curr->next = head_hi; curr = curr->next;
    }

    auto res = ghost->next; 
    ghost->next = NULL; delete ghost;
    return res;
}


void TEST(vector<int> arr) {
    auto head = new ListNode(arr);
    head->print();
    auto res = sortList(head);
    res->print();
}

int main() {
    TEST({1,3,4,2,7,6,5});
}

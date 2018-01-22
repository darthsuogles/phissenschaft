#include <iostream>
#include <vector>
#include "ListNode.hpp"

using namespace std;

typedef LnkLstNode<int> ListNode;

ListNode *detectCycle(ListNode *head) {
    if (NULL == head) return NULL;
    auto ptr1 = head, ptr2 = head;
    // First find an intersection point of the two
    // This might not be the loop initial node
    do { 
        if (NULL == ptr1) return NULL;
        ptr1 = ptr1->next;
        if (NULL == ptr2) return NULL;
        if (NULL == ptr2->next) return NULL;
        ptr2 = ptr2->next->next;
    } while (ptr1 != ptr2);

    // Then if there is a cycle
    // l + r = n
    // d = l + 2d (mod r)  =>  d = d + (l + d) (mod r)
    // thus d + l = 0 (mod r)
    auto fndr = head;
    for (; fndr != ptr1; ptr1 = ptr1->next, fndr = fndr->next);
    return fndr;
}

void TEST(vector<int> arr) {
    auto head = new ListNode(arr);
    auto head_cycle = detectCycle(head);
    if (NULL != head_cycle) 
        cout << head_cycle->val << endl;
    delete head;
}

int main() {
    TEST({1,2,3});
}

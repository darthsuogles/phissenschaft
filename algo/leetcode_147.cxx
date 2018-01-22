#include <iostream>
#include "ListNode.hpp"

using namespace std;

typedef LnkLstNode<int> ListNode;

ListNode* insertionSortList(ListNode *head) {
    if (NULL == head) return NULL;

    auto ghost = new ListNode(INT_MIN);
    ghost->next = head;
    auto node = head, last = ghost;
    while (NULL != node) {
        auto curr = ghost->next, prev = ghost;
        while (node != curr) {
            if (curr->val >= node->val)
                break;
            prev = curr;
            curr = curr->next;
        }
        auto tmp = node->next;
        if (node != curr) {
            last->next = tmp;
            prev->next = node;
            node->next = curr;
        } else {
            last = node;
        }
        node = tmp;
    }   
    last->next = NULL;
    return ghost->next;
}

ListNode* insertionSortListRec(ListNode *head) {
    if (NULL == head) return NULL;
    if (NULL == head->next) return head;

    auto sorted = insertionSortListRec(head->next);
    if (head->val <= sorted->val) {
        head->next = sorted;
        return head;
    }    
        
    auto curr = sorted, prev = sorted;
    while (NULL != curr) {
        if (curr->val >= head->val)
            break;
        prev = curr;
        curr = curr->next;
    }
    prev->next = head;
    head->next = curr;
    return sorted;
}


void TEST(vector<int> nums) {
    if (nums.empty()) {
        cout << "Ok" << endl; return;
    }
    auto head = new ListNode(nums);
    head->print();
    auto res = insertionSortList(head);
    res->print();
    bool valid = true;
    int cnt = 0;
    auto curr = head;
    for (; NULL != curr; ++cnt, curr = curr->next);
    for (; NULL != res->next; res = res->next, --cnt) {
        if (res->val > res->next->val) {
            valid = false; break;
        }
    }
    valid = valid && (1 == cnt);
    if (!valid)
        cout << "Error";
    else
        cout << "Ok";
    cout << endl;
}

int main() {
    TEST({1,2,5,7,4,2,5});
    TEST({3,2,1});    
}

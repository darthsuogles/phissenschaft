#include <iostream>
#include "ListNode.hpp"

using namespace std;

typedef LnkLstNode<int> ListNode;

ListNode* rev(ListNode *head) {
    if (NULL == head) return NULL;
    if (NULL == head->next) return head;
    auto nxt_old = head->next;
    auto nxt_new = rev(head->next);
    nxt_old->next = head;
    head->next = NULL;
    return nxt_new;
}

bool isPalindrome(ListNode *head) {
    if (NULL == head) return true;
    if (NULL == head->next) return true;
    int len = 0;
    auto tail = head;
    for (; NULL != tail->next; ++len, tail = tail->next);
    ++len;
    auto node_mid = head;
    for (int i = 0; i < len/2; ++i, node_mid = node_mid->next);
    if (1 == len % 2) node_mid = node_mid->next;

    auto na = head, nb = rev(node_mid);
    for (int i = 0; i < len/2; ++i) {
        if (na->val != nb->val) 
            break;
        na = na->next;
        nb = nb->next;
    }
    bool res = NULL == nb;
    for (na = head; na->next != node_mid; na = na->next);
    na->next = rev(tail);
    return res;
}

void TEST(vector<int> arr) {
    auto head = new ListNode(arr);
    cout << "AVANT: "; head->print();    
    cout << isPalindrome(head) << endl;
    cout << "APRES: "; head->print();
    cout << "-------------" << endl;
    delete head;
}

int main() {
    TEST({1,2,3,4,5});    
    TEST({1, 2});
    TEST({1});
    TEST({0, 0});
    TEST({});
}


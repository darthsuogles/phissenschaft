#include <iostream>
#include <vector>
#include "ListNode.hpp"

using namespace std;

typedef LnkLstNode<int> ListNode;

// Return the carry
int incr(ListNode *l, int a, int n) {
    if (0 == n) return 0;
    if (NULL == l) return 0;
    if (0 == a) return 0;

    int carry = (1 == n) ? a : incr(l->next, a, n-1);
    if (carry > 0) {
        int sum = l->val + carry;
        l->val = sum % 10;
        return sum / 10;
    } 
    return 0;
}

int length(ListNode *l) {
    int n = 0;
    for (; NULL != l; ++n, l = l->next);
    return n;
}

ListNode* addEq(ListNode *l1, ListNode *l2, int n, int& carry) {
    if (0 == n) return NULL;
    int a = 0;
    auto res = addEq(l1->next, l2->next, n-1, a);
    int sum = l1->val + l2->val + a;
    
    carry = sum / 10;
    ListNode *root = new ListNode(sum % 10);
    root->next = res;
    return root;
}

ListNode* addTwoNumbers(ListNode *l1, ListNode *l2) {
    if (NULL == l1) return l2;
    if (NULL == l2) return l1;
    int len1 = length(l1);
    int len2 = length(l2);
    if (len1 < len2) {
        ListNode *l_tmp = l1;
        l1 = l2;
        l2 = l_tmp;
        int len_tmp = len1;
        len1 = len2;
        len2 = len_tmp;
    } 

    auto l0 = l1;
    for (int d = len1 - len2; d > 0; l0 = l0->next, --d);
    
    int carry = 0;
    auto res_next = addEq(l0, l2, len2, carry);
    ListNode *tmp, *curr = res_next;

    if (len1 != len2) {
        curr = tmp = new ListNode(-1);
        for (int d = len1 - len2; d > 0; --d) {
            curr->next = new ListNode(l1->val);
            l1 = l1->next;
            curr = curr->next;
        }
        curr->next = res_next; 
        curr = tmp->next;
        tmp->next = NULL; delete tmp;
        
        carry = incr(curr, carry, len1 - len2);    
    }

    if (carry > 0) {
        tmp = curr;
        curr = new ListNode(carry);
        curr->next = tmp;
    }
    return curr;
}

int toNum(ListNode *l) {
    if (NULL == l) return 0;    
    int res = 0;
    for (; NULL != l; l = l->next) res = res * 10 + l->val;
    return res;
}

void TEST(vector<int> arr1, vector<int> arr2, int tgt) {
    cout << "-----" << endl;
    auto l1 = new ListNode(arr1);
    auto l2 = new ListNode(arr2);
    auto res_lst = addTwoNumbers(l1, l2);
    auto res = toNum(res_lst);
    if (tgt == res) 
        cout << "OK" << endl;
    else {
        l1->print(); l2->print();
        cout << "ERR " << tgt << " != " << res << endl;
    }
    delete res_lst; delete l1; delete l2;
}

int main() {
    TEST({5}, {5}, 10);
    TEST({1}, {9, 9}, 100);
    TEST({9, 9}, {1}, 100);
    TEST({7,2,4,3}, {5,6,4}, 7807);
}

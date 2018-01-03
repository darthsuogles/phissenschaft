/**
 * LeetCode Problem 92
 *
 * Reverse linked list II
 */

#include <iostream>

using namespace std;

struct ListNode {
  int val;
  ListNode *next;
  ListNode(int x): val(x), next(NULL) {}
};

class Solution {
  void reverse_list(ListNode *head, ListNode *tail)
  {
    if ( ! head || ! tail ) return;
    if ( head == tail ) return;

    ListNode *tl = head->next;
    ListNode *last = tail->next;
    reverse_list(head->next, tail);
    head->next = last;
    tl->next = head;
  }
  
public:
  ListNode* reverseBetween(ListNode* head, int m, int n)
  {
    if ( NULL == head ) return NULL;
    
    ListNode *hd = head, *tl = head;
    ListNode *pp = new ListNode(-1); pp->next = head;    
    ListNode *prev = pp;
    int i = 1; // list index one based
    for (; i < m; ++i, prev = hd, hd = hd->next, tl = tl->next);
    for (; i < n; ++i, tl = tl->next);
    
    ListNode *last = tl->next;
    reverse_list(hd, tl);
    prev->next = tl;
    hd->next = last;

    head = pp->next;
    delete pp;
    return head;
  }
};

int main()
{
  Solution sol;
  int A[] = {1,2,3,4,5};
  int len = sizeof(A) / sizeof(int);
  ListNode *head = new ListNode(A[0]);
  ListNode *curr = head;
  for (int i = 1; i < len; ++i)
    {
      curr->next = new ListNode(A[i]);
      curr = curr->next;
    }

  ListNode *new_head = sol.reverseBetween(head, 2, 4);
  curr = new_head;
  for (int i = 0; i < len; ++i, curr = curr->next)
    cout << curr->val << " ";
  cout << endl;
}

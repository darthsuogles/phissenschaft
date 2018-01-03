#include <iostream>

using namespace std;

struct ListNode {
  int val;
  ListNode *next;
  ListNode(int x) : val(x), next(NULL) {}
};
  
class Solution {
public:
  ListNode *mergeTwoLists(ListNode *l1, ListNode *l2)
  {
    if ( NULL == l1 )
      return l2;
    else if ( NULL == l2 )
      return l1;

    ListNode *head = new ListNode(0); // temporary 
    ListNode *curr = head;
    while ( l1 != NULL && l2 != NULL )
      {
	int a = l1->val, b = l2->val;
	if ( a < b )
	  {
	    curr->next = l1;
	    l1 = l1->next;
	  }
	else
	  {
	    curr->next = l2;
	    l2 = l2->next;
	  }
	if ( NULL == head )
	  head = curr;
	curr = curr->next;
      }

    if ( NULL == l1 )
      curr->next = l2;
    else if ( NULL == l2 )
      curr->next = l1;

    return head->next;
  }
};

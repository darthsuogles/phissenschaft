/**
 * LeetCode Problem 24
 *
 * Swap nodes in pairs 
 */

#include <iostream>
#include <vector>

using namespace std;

struct ListNode {
  int val;
  ListNode *next;
  ListNode(int x) : val(x), next(NULL) {}
};

class Solution {
public:
  ListNode *swapPairs(ListNode *head)
  {
    if ( NULL == head ) return NULL;
    
    ListNode *nlst = new ListNode(0);
    nlst->next = head;
    ListNode *prev = nlst;
    ListNode *curr = head; // swap curr with curr->next
    while ( curr != NULL )
      {
	ListNode *nxt = curr->next;
	if ( NULL == nxt )
	  break;
	prev->next = nxt;
	curr->next = nxt->next;
	nxt->next = curr;

	// Move two forward
	prev = curr;
	curr = curr->next;	
      }

    curr = nlst->next;
    delete nlst; // clean up the memory
    return curr;
  }
};

ListNode *build_list(int A[], int len)
{
  if ( 0 == len ) return NULL;
  
  ListNode *head = new ListNode(A[0]);
  ListNode *curr = head;
  for (int i = 1; i < len; ++i)
    {
      curr->next = new ListNode(A[i]);
      curr = curr->next;
    }

  return head;
}

void print_list(ListNode *head)
{
  while ( head != NULL )
    {
      cout << head->val << " -> ";
      head = head->next;
    }
  cout << "NIL" << endl;
}

int main()
{
  int A[] = {1,2,3,4};
  int len = sizeof(A) / sizeof(int);

  ListNode *head = build_list(A, len);
  Solution sol;
  print_list(head);
  ListNode *new_head = sol.swapPairs(head);
  print_list(new_head);
}

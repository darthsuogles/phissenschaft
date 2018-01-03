/**
 * LeetCode Problem 25
 *
 * Reverse linked list within k group at a time
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

  void reverse_list(ListNode *head, ListNode *last,
		    ListNode **new_head, ListNode **new_last)
  {
    if ( head == last || head->next == NULL )
      {
	*new_head = *new_last = head;
	return;
      }

    reverse_list(head->next, last, new_head, new_last);

    head->next = NULL;
    (*new_last)->next = head;
    (*new_last) = head;
  }
  
public:
  ListNode *reverseKGroup(ListNode *head, int k)
  {
    if ( 1 == k ) return head;
    if ( NULL == head ) return NULL;
    
    ListNode *last = head;

    int cnt = 1;
    for (; cnt < k; ++cnt)
      {
	if ( last->next != NULL )
	  last = last->next;
	else
	  break;
      }
    if ( cnt < k ) // the list has less than k items
      return head;

    ListNode *new_head, *new_last;
    ListNode *nxt = last->next;
    reverse_list(head, last, &new_head, &new_last);    
    new_last->next = reverseKGroup(nxt, k);
    return new_head;
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
  int A[] = {1,2,3,4,5,6,7};
  int len = sizeof(A) / sizeof(int);

  ListNode *head = build_list(A, len);
  Solution sol;
  print_list(head);
  ListNode *new_head = sol.reverseKGroup(head, 3);
  print_list(new_head);
}


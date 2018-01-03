#include <iostream>

using namespace std;

struct ListNode {
  int val;
  ListNode *next;
  ListNode(int x) : val(x), next(NULL) {}

  ListNode *gen_nodes(int A[], const int len) { return gen_nodes(A, 0, len); }
  ListNode *gen_nodes(int A[], int pos, const int len)
  {
    if ( len == pos )
      return NULL;

    ListNode *curr = new ListNode(A[pos]);
    curr->next = gen_nodes(A, pos+1, len);
    return curr;
  }  
};


class Solution {
public:
  ListNode *removeNthFromEnd(ListNode *head, int n)
  {
    if ( NULL == head )
      return NULL;

    int cnt = 0;
    ListNode *curr = head;
    while ( curr )
      {
	curr = curr->next;
	++cnt;
      }
        
    int bnd = cnt - n; 
    if ( 0 == bnd )
      {
	ListNode *res = head->next;
	delete head;
	return res;
      }

    ListNode *target = head->next;
    ListNode *prev = head;
    for ( int i = 1; i < bnd; ++i )
      {
	prev = target;
	target = target->next;
      }
        
    if ( target != NULL )
      {
	prev->next = target->next;
	delete target;
      }
    return head;
  }
};

int main()
{
  Solution sol;

  ListNode *head = new ListNode(1);
  head = sol.removeNthFromEnd(head, 1);
  while ( head != NULL )
    {
      cout << head->val << " ";
      head = head->next;
    }
  cout << endl;

  
}

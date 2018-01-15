/**
 * LeetCode Problem 23
 *
 * Merge k sorted lists
 */

#include <iostream>
#include <vector>
#include <queue>

using namespace std;

struct ListNode {
  int val;
  ListNode *next;
  ListNode(int x) : val(x), next(NULL) {}
};

class ListNodeComparison {
public:
  // Implements the greater operator
  bool operator() (const ListNode *lhs, const ListNode *rhs)
  {
    return ( n1->val > n2->val );
  }
};

class Solution { 
public:
  ListNode *mergeKLists(vector<ListNode *> &lists)
  {
    //priority_queue< int, vector<int>, greater<int> > min_heap;
    priority_queue< ListNode*, vector<ListNode*>, ListNodeComparison > min_heap;
    int K = lists.size();
    for (int i = 0; i < K; ++i)
      {
	if ( lists[i] != NULL ) // the node is never NULL
	  min_heap.push( lists[i] );
      }

    ListNode *head = new ListNode(0);
    ListNode *curr = head;
    while ( ! min_heap.empty() )
      {
	// Find the min element and its associated list
	ListNode *nxt = min_heap.top();
	min_heap.pop();
	if ( nxt->next != NULL ) // the node is never NULL
	  min_heap.push( nxt->next );
	curr->next = nxt;
	curr = curr->next;
      }

    return head->next;
  }
};
    
  
    

      

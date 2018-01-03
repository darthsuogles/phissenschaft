/**
 * LeetCode Problem 86
 *
 * https://leetcode.com/problems/partition-list/
 */

#include <iostream>

using namespace std;

/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */

struct ListNode {
  int val;
  ListNode *next;
  ListNode(int x): val(x), next(NULL) {}
};

class Solution {
public:
    ListNode* partition(ListNode* head, int x) {
        if ( NULL == head ) return NULL;
        ListNode *ghost = new ListNode(-1);
        ghost->next = head;
        ListNode *prev = ghost;
        ListNode *curr = head;
        ListNode *ghost_geq_x = new ListNode(-1);
        ListNode *curr_geq_x = ghost_geq_x;
        while ( NULL != curr ) {
            if ( curr->val >= x ) {
                prev->next = curr->next;
                curr->next = NULL;
                curr_geq_x->next = curr;
                curr_geq_x = curr;
            } else {
                prev = curr;
            }
	    curr = prev->next;
        }
        prev->next = ghost_geq_x->next;
        head = ghost->next;
        delete ghost;
        delete ghost_geq_x;
        return head;
    }
};

void delete_list(ListNode *head) {
  if ( NULL == head ) return;
  delete_list(head->next);
  head->next = NULL;
  delete head;
}

void test_case(Solution &sol, int A[], int len, int x) {
  ListNode *head = new ListNode(A[0]); 
  ListNode *curr = head;	       
  for (int i = 1; i < len; ++i)	
    curr->next = new ListNode(A[i]);  
  ListNode *res = sol.partition(head, x);
  curr = res;
  while ( NULL != curr ) {
    cout << curr->val << " ";
    curr = curr->next;
  }
  cout << endl;
  delete_list(res);
}

int main() {
  ListNode *head = new ListNode(1);
  int A[] = {1, 2};
  Solution sol;
  test_case(sol, A, 2, 2);
}

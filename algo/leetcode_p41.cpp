/**
 * LeetCode Problem 40
 *
 * First missing positive 
 */

#include <iostream>
#include <limits.h>
#include <set>

using namespace std;

class Solution {

  // Find the first missing integer greater than a
  int missing_int_gt(int a, int A[], int p, int q)
  {
    if ( p > q ) return a+1;
    if ( p == q )
      {
	if ( A[p] != a + 1 )
	  return a+1;
	else
	  return a+2;
      }
    
    int pivot = A[q];
    int i = p, j = q-1;
    while ( i < j )
      {
	while ( A[i] < pivot ) ++i;
	while ( A[j] > pivot && j > p ) --j;

	if ( i < j )
	  {
	    int tmp = A[i]; A[i] = A[j]; A[j] = tmp;
	    ++i; --j;
	  }
      }
    if ( A[i] > pivot )
      { int tmp = A[i]; A[i] = A[q]; A[q] = tmp; }

    if ( pivot <= a )
      return missing_int_gt(a, A, i+1, q);
    
    set<int> tbl;
    for (int k = p; k <= j; ++k) // assuming unique integers
      if ( a < A[k] && A[k] < pivot ) tbl.insert(A[k]);
    int cnt = tbl.size();
    if ( 0 == cnt )
      {
	if ( a+1 < pivot )
	  return a+1;
	else
	  return missing_int_gt(pivot, A, i+1, q);
      }
    else if ( (a+1) + cnt < pivot )
      return missing_int_gt(a, A, p, j);
    else
      return missing_int_gt(pivot, A, i+1, q);
  }
  
public:
  int firstMissingPositive(int A[], int n)
  {
    if ( 0 == n ) return 1;
    int res = missing_int_gt(0, A, 0, n-1);
    if ( -1 == res )
      return 0;
    else
      return res;
  }
};

int main()
{
  Solution sol;
  
#define test_case(A) {					\
    int len = sizeof(A) / sizeof(int);			\
    int res = sol.firstMissingPositive(A, len);		\
    cout << "ans: " << res << endl;			\
  }							\
  
  int A[] = {2,1};
  test_case(A);
  int B[] = {4,3,4,1,1,4,1,4};
  test_case(B);
  int C[] = {0,1,2,3,4};
  test_case(C);
}

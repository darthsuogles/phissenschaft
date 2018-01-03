/**
 * LeetCode Problem 81
 *
 * Search in rotated sorted array, with duplicates
 */

#include <iostream>
#include <algorithm>

using namespace std;

class Solution {
public:
  bool search(int A[], int n, int target)
  {
    if ( n <= 0 ) return false;
    if ( 1 == n ) return (target == A[0]);
    if ( 2 == n ) return (target == A[0] || target == A[1]);

    int i = 0, j = n-1;
    while ( i + 1 < j )
      {	
	int a = A[i], b = A[j];
	if ( a == target || b == target ) return true;
	
	if ( a < b ) // just a binary search
	  return binary_search(A+i, A+j+1, target);
	else if ( a > b ) // in between the rotation
	  {
	    int k = (i+j)/2;
	    int mval = A[k];
	    if ( mval == target )
	      return true;
	    if ( mval < a ) // a ... hi ... mval ... b
	      {
		if ( target < mval )
		  j = k;
		else if ( target < b )
		  return binary_search(A+k, A+j+1, target);
		else if ( target < a )
		  return false;
		else
		  j = k;
	      }
	    else if ( mval > a ) // a ... mval ... hi ... b
	      {
		if ( target < b )
		  i = k;
		else if ( target < a )
		  return false;
		else if ( target < mval )
		  return binary_search(A+i, A+k+1, target);
		else
		  i = k;
	      }
	    else // a ... (mval = a) ... hi ... b
	      i = k;
	  }
	else // a == b
	  {
	    int k = (i+j)/2;
	    int mval = A[k];
	    if ( mval == target )
	      return true;
	    if ( mval < a ) // a ... hi ... mval ... (b = a)
	      {
		if ( target < mval )
		  j = k;
		else if ( target < b )
		  return binary_search(A+k, A+j+1, target);
		else
		  j = k;
	      }
	    else if ( mval > a ) // a ... mval ... hi ... (b = a)
	      {
		if ( target < b )
		  i = k;
		else if ( target < mval )
		  return binary_search(A+i, A+k+1, target);
		else
		  i = k;
	      }
	    else // a ... (mval = a) ... (b = a)
	      {
		if ( search(A+i, k-i, target) )
		  return true;
		return ( search(A+k, j-k+1, target) );
	      }
	  }
      }

    return (target == A[i] || target == A[j]);
  }
};

int main()
{
  Solution sol;

#define test_case(A, target, valid)  {					\
    int len = sizeof((A)) / sizeof(int);				\
    bool res = sol.search((A), len, (target));				\
    if ( res == valid ) cout << "Passed" << endl; else cout << "Failed" << endl; \
  }									\
  
  int A[] = {1,2,1};
  test_case(A, 2, true);
  test_case(A, 1, true);

  int B[] = {4,5,6,7,0,1,2};
  test_case(B, 0, true);
  test_case(A, 2, true);
}

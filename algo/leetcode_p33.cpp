/**
 * LeetCode Problem 33
 *
 * Search in a pivot-rotated sorted array
 */

#include <iostream>

using namespace std;

class Solution {
public:
  int search(int A[], int n, int target)
  {
    int i = 0, j = n-1;
    while ( i + 1 < j )
      {
	int a = A[i], b = A[j];
	if ( a < b )
	  {
	    // Just a normal binary search
	    int k = (i+j)/2;
	    int mid = A[k];
	    if ( mid < target )
	      i = k;
	    else if ( mid == target )
	      return k;
	    else
	      j = k;
	  }
	else if ( a > b )
	  {
	    // The reverse pivot is in-between
	    if ( b < target && target < a )
	      {
		int k = n-1;
		// Becomes normal binary search
		if ( target < A[k] ) // A[0] .. a .. pivot .. b .. target .. A[n-1]
		  {
		    i = j; j = k;
		  }
		else if ( target == A[k] )
		  return k;
		else // A[0] .. target .. a .. pivot .. b .. A[n-1]
		  {
		    if ( target < A[0] ) return -1;
		    j = i; i = 0;
		  }
	      }
	    else if ( b == target )
	      return j;
	    else if ( a == target )
	      return i;
	    else if ( a < target ) // a ... target ... pivot ... b
	      {
		int k = (i+j)/2;
		int mid = A[k];
		if ( mid < b ) // a .. target .. pivot .. mid .. b
		  j = k;
		else if ( mid == target )
		  return k;
		else if ( mid < target ) // a .. mid .. target .. pivot .. b
		  i = k;
		else // a .. target .. mid .. pivot
		  j = k; 
	      }
	    else // a ... pivot ... target ... b
	      {
		int k = (i+j)/2;
		int mid = A[k];
		if ( target == mid )
		  return k;
		else if ( mid < target )
		  i = k;
		else if ( mid < b ) // a .. pivot .. target .. mid .. b
		  j = k;
		else // a .. mid .. pivot .. target .. b
		  i = k;
	      }
	  }
	else // a == b
	  {
	    if ( target != a )
	      return -1;
	    else
	      return i;
	  }
      }
    if ( A[i] == target )
      return i;
    else if ( A[j] == target )
      return j;
    else
      return -1;
  }
};

int main()
{
  Solution sol;

  int A[] = {4,5,6,7,0,1,2};
  int len = sizeof(A) / sizeof(int);
  for (int i = 0; i < len; ++i)
    cout << (sol.search(A, len, A[i]) == i) << endl;
  cout << (sol.search(A, len, 11) == -1) << endl;
  
  int B[] = {5,1,2,3,4};
  len = sizeof(B) / sizeof(int);
  for (int i = 0; i < len; ++i)
    cout << (sol.search(B, len, B[i]) == i) << endl;
  cout << (sol.search(B, len, 17) == -1) << endl;
}

/**
 * LeetCode Problem 26
 *
 * Remove duplicates
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <limits.h>

using namespace std;

class Solution {
public:
  int removeDuplicates(int A[], int n)
  {
    if ( 0 == n ) return 0;
    //int cnt = 0;
    int idx = 0;
    for (int i = 1; i < n; ++i)
      {
	if ( A[i] != A[i-1] )
	  {
	    //A[i] = INT_MAX;
	    //++cnt;
	    A[++idx] = A[i];
	  }
      }
    //sort(A, A+n);
    //return n - cnt;
    return idx+1;
  }
};

int main()
{
  int A[] = {1,2,5,5,6,7,7,8,9,9};
  int len = sizeof(A) / sizeof(int);

  Solution sol;
  int n = sol.removeDuplicates(A, len);
  for (int i = 0; i < n; ++i)
    cout << A[i] << " ";
  cout << endl;
}

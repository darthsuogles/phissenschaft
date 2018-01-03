/**
 * LeetCode Problem 77
 *
 * Remove triplicates ...
 */

#include <iostream>

using namespace std;

class Solution {
public:
  int removeDuplicates(int A[], int n)
  {
    if ( n <= 2 ) return n;
        
    int j = 0;
    for (int i = 0; i + 2 < n; ++i)
      {
	if ( (A[i] != A[i+1]) || (A[i] != A[i+2]) )
	  A[j++] = A[i];
      }
    for (int i = n-2; i < n; A[j++] = A[i++]);
    
    return j;
  }
};

int main()
{
  Solution sol;

#define test_case(A) {				\
    int len = sizeof(A) / sizeof(int);		\
    int nlen = sol.removeDuplicates(A, len);	\
    for (int i = 0; i < nlen; ++i)		\
      cout << A[i] << " ";			\
    cout << endl; }				\
  
  int A[] = {1,1,1,2,2,3};
  test_case(A);
}

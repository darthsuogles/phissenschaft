/**
 * LeetCode Problem 75
 *
 * Sort array with 3 values
 */

#include <iostream>

using namespace std;

class Solution {
public:
  void sortColors(int A[], int n)
  {
    if ( n < 2 ) return;
    int i = 0;
    for (int pivot = 0; pivot < 2; ++pivot)
      {
	int j = n-1;
	while ( i < j )
	  {
	    for (; i < n; ++i)
	      if ( A[i] > pivot ) break;
	    if ( i >= n ) break;

	    for (; j >= 0; --j)
	      if ( A[j] <= pivot ) break;
	    if ( j < 0 ) break;
	    
	    if ( i < j )
	      {
		int tmp = A[i];
		A[i++] = A[j];
		A[j--] = tmp;
	      }
	  }
      }
  }
};

int main()
{
  Solution sol;

#define test_case(A)  {				\
    int len = sizeof((A)) / sizeof(int);	\
    sol.sortColors((A), len);			\
    for (int i = 0; i < len; ++i)		\
      cout << (A)[i] << " ";			\
    cout << endl;				\
  }						\

  int A[] = {1,1,2,2,2,1,1,0,0,1,1,2,2};
  test_case(A);

  int B[] = {1,0};
  test_case(B);

  int C[] = {1,1,0,0,0,1,1,1};
  test_case(C);
}
	

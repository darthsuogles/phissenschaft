/**
 * LeetCode Problem 53
 *
 * Maximum contiguous subarray
 */

#include <iostream>
#include <limits.h>

using namespace std;

class Solution {
public:
  int maxSubArray(int A[], int n)
  {
    if ( 0 == n ) return 0;
    int max_sum = A[0];
    int curr_sum = 0;
    for (int i = 0; i < n; ++i)
      {
	if ( curr_sum < 0 ) curr_sum = 0;
	curr_sum += A[i];
	if ( curr_sum > max_sum )
	  max_sum = curr_sum;
      }
    return max_sum;
  }
};

int main()
{
  Solution sol;
  int A[] = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
  //int A[] = {-2, -1, 1};
  int len = sizeof(A) / sizeof(int);
  cout << sol.maxSubArray(A, len) << endl;
}


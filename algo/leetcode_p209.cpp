/**
 * LeetCode Problem 209
 *
 * Minimum size subarray sum
 */

#include <vector>
#include <iostream>

using namespace std;

class Solution {
public:
  int minSubArrayLen(int s, vector<int>& nums)
  {
    if ( nums.empty() ) return 0;
    int n = nums.size();
    int i = 0, j = 0;
    int min_len = n + 1;

    int cnt = 0;
    while ( j < n )
      {
	for (; j < n; cnt += nums[j++])
	  if ( cnt >= s ) break;
	if ( cnt < s ) break;
	
	int curr_len = j - i + 1;
	for (; i < j; cnt -= nums[i++])
	  {
	    if ( cnt < s ) break;
	    --curr_len; 
	  }
	min_len = min( curr_len, min_len );
      }

    if ( n + 1 == min_len ) return 0;
    return min_len;
  }
};

int main()
{
  Solution sol;

#define test_case(s, A)  {				\
    int len = sizeof((A)) / sizeof(int);		\
    vector<int> vec((A), (A) + len);			\
    cout << sol.minSubArrayLen((s), vec) << endl;  }	\

  int A[] = {2,3,1,2,4,3};
  test_case(7, A);

  int B[] = {1,1};
  test_case(3, B);

  int C[] = {50, 1, 2};
  test_case(4, C);
}

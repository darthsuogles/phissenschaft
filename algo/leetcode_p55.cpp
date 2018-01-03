/**
 * LeetCode Problem 55
 *
 * Jump game
 */

#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
  // Recursive solution
  bool canJump_v0(int A[], int n)
  {
    if ( n <= 1 ) return true;
    for (int j = 1; j <= A[0]; ++j)
      if ( canJump(A + j, n - j) ) return true;
    return false;
  }

  // Dynamic programming 
  bool canJump_v1(int A[], int n)
  {
    vector<bool> tbl(n, false);
    tbl[n-1] = true;
    for (int i = n-2; i >= 0; --i)
      {
	for (int j = A[i]; j >= 1 && i + j < n; --j)
	  if ( tbl[i + j] )
	    {
	      tbl[i] = true;
	      break;
	    }
      }

    return tbl[0];
  }

  bool canJump(int A[], int n)
  {
    if ( n <= 1 ) return true;
    int p = 0, q = A[0];
    while ( p < q )
      {
	int max_pos = q;
	for (int i = p+1; i <= q; ++i)
	  {
	    int curr = i + A[i];
	    if ( curr >= n-1 )
	      return true;
	    if ( curr > max_pos )
	      max_pos = curr;
	  }
	p = q;
	q = max_pos;
      }

    return false;
  }
};

int main()
{
  Solution sol;
#define test_case(A) cout << sol.canJump((A), sizeof((A)) / sizeof(int)) << endl;
  int A[] = {2,3,1,1,4};
  test_case(A);
  int B[] = {3,2,1,0,4};
  test_case(B);
  int C[] = {5,9,3,2,1,0,2,3,3,1,0,0};
  test_case(C);
  int D[] = {1,1,1,0};
  test_case(D);
}

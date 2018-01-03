/**
 * LeetCode Problem 42
 *
 * Trapping rain water
 */

#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
  int trap(int A[], int n)
  {
    if ( n <= 2 ) return 0;
    vector<int> max_left(n, 0);
    vector<int> max_right(n, n-1);

    for (int i = 1, j = n-2; i < n; ++i, --j)
      {
	int p = max_left[i-1];
	max_left[i] = (A[p] > A[i-1]) ? p : (i-1);
	int q = max_right[j+1];
	max_right[j] = (A[q] > A[j+1]) ? q : (j+1);
      }

    int init = 0; // candidate for the left wall
    int vol_tot = 0;
    while ( init < n )
      {
	//cout << init << endl;
	int rw = max_right[init]; // the right-most wall
	if ( rw == init ) break; // hitting the right boundary
	int next = rw;
	int lw = max(init, max_left[rw]); // don't want to back track
	//printf("lw: %d, rw: %d (outside)\n", lw, rw);
	
	while ( lw >= init )
	  {
	    //printf("lw: %d, rw: %d\n", lw, rw);
	    int vol = 0;
	    int level = min(A[rw], A[lw]);
	    for (int k = lw+1; k < rw; ++k)
	      vol += level - A[k];
	    vol_tot += vol;

	    rw = lw;
	    lw = max(init, max_left[lw]);
	    if ( lw == rw ) break; // hitting the left boundary
	  }

	init = next;
      }

    return vol_tot;
  }
};

int main()
{
  Solution sol;
  int A[] = {0,1,0,2,1,0,1,3,2,1,2,1};
  int len = sizeof(A) / sizeof(int);

  int res = sol.trap(A, len);
  cout << "ans: " << res << endl;
}

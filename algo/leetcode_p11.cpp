/**
 * LeetCode Problem 11: container with most water
 */
#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
  int maxArea(vector<int> &height)
  {
    int len = height.size();
    int max_vol = 0;
    int i = 0, j = len - 1;
    while ( i < j )
      {
	int hi = height[i], hj = height[j];
	if ( hi < hj )
	  {
	    // The area is limited by the smaller of the two.
	    // Any other configuration involving this index
	    // will result in an even smaller area, thus not considered.
	    // Apply induction on indices within the range, for the
	    // indices outside (i, j) are already considered.
	    max_vol = max(max_vol, (j-i) * hi);
	    ++i;
	  }
	else
	  {
	    max_vol = max(max_vol, (j-i) * hj);
	    --j;
	  }
      }
        
    return max_vol;
  }  
  
  /**
   * This solves a more difficult problem
   */
  int maxAreaJag(vector<int> &height)
  {
    int len = height.size();
    vector<int> max_left_inds(len);
    vector<int> max_right_inds(len);

    int mxl_idx = -1, mxr_idx = -1;
    int mxl = -1, mxr = -1;
    for (int i = 0; i < len; ++i)
      {
	int curr = height[i];
	if ( curr > mxl )
	  {
	    mxl = curr;
	    max_left_inds[i] = mxl_idx;
	    mxl_idx = i;
	  }
	else
	  max_left_inds[i] = mxl_idx;
      }
    for (int j = len-1; j >= 0; --j)
      {
	int curr = height[j];
	if ( curr > mxr )
	  {
	    mxr = curr;
	    max_right_inds[j] = mxr_idx;
	    mxr_idx = j;
	  }
	else
	  max_right_inds[j] = mxr_idx;
      }

    // Main loop
    int max_vol = 0;
    int i = 0;
    for (; i < len;)
      {
	int bar_r = max_right_inds[i];
	if ( bar_r == -1 ) // at the right-most position
	  break; 
	if ( bar_r == i ) // hit the wall, move to the right
	  {
	    ++i;
	    continue;
	  }
	else
	  i = bar_r;	    
	
	int bar_l = max_left_inds[bar_r];
	if ( bar_r - bar_l == 1 )
	  continue;

	// Compute the volume
	int top = min(height[bar_r], height[bar_l]);
	int vol = 0;
	for (int k = bar_l + 1; k < bar_r; ++k)
	  vol += top - height[k];
	if ( vol > max_vol )
	  max_vol = vol;
      }

    return max_vol;
  }
};

int main()
{
  Solution sol;
  // #define test_case(Ai) {				\
  //     int A[] = {Ai};				\
  //     cout << sol.maxArea(A) << endl;		\
  //   }						\

  int A[] = {1,4,3,2,5};
  vector<int> vA(A, A + sizeof(A) / sizeof(int));
  cout << sol.maxArea(vA) << endl;

  const int len = 15000;
  int B[len];
  for (int i = 0; i < len; B[i] = len - i, ++i);
  vector<int> vB(B, B + len);
  cout << sol.maxArea(vB) << endl;
}

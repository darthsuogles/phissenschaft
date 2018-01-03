/**
 * LeetCode Problem 84
 *
 * Largest rectangle in histogram
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <climits>

using namespace std;

class Solution {    
public:
  /**
   * Using range minimum query
   */
  int largestRectangleArea(vector<int> &height) {
    if ( height.empty() ) return 0;

    // O(n^2) time and space solution
    // For better solution
    // Ref: https://www.topcoder.com/community/data-science/data-science-tutorials/range-minimum-query-and-lowest-common-ancestor/#Range_Minimum_Query_(RMQ)
    size_t len = height.size();
    int gMaxVal = 0;
    vector< vector<int> > tblRngMin(len, vector<int>(len, INT_MAX));
    for (int i = 0; i < len; ++i)
      gMaxVal = max(gMaxVal, tblRngMin[i][i] = height[i]);
    for (int i = 0; i+1 < len; ++i) {
      for (int j = i+1; j < len; ++j) {
	int curr;
	if ( tblRngMin[i][j-1] > height[j] ) {
	  curr = tblRngMin[i][j] = height[j];
	} else {
	  curr = tblRngMin[i][j] = tblRngMin[i][j-1];
	}
	gMaxVal = max(curr * (j-i+1), gMaxVal);
      }
    }
    return gMaxVal;
  }
};

int main() {
  Solution sol;
  vector<int> height = {2,1,5,6,2,3};
  cout << sol.largestRectangleArea(height) << endl;
}

/**
 * LeetCode Problem 63
 *
 * Unique paths with obstacles
 */

#include <iostream>

using namespace std;

class Solution {
public:
  int uniquePathsWithObstacles(vector<vector<int> > &obstacleGrid)
  {
    int m = obstacleGrid.size();
    if ( 0 == m ) return 1;
    int n = obstacleGrid[0].size();
    if ( 0 == n ) return 1;
    if ( 1 == obstacleGrid[m-1][n-1] ) return 0;
    
    vector< vector<int> > tbl(m, vector<int>(n, 0));
    tbl[m-1][n-1] = 1;
    for (int i = m-2; i >= 0; --i)
      {
	if ( 1 == obstacleGrid[i][n-1] )
	  tbl[i][n-1] = 0;
	else
	  tbl[i][n-1] = tbl[i+1][n-1];
      }
    for (int j = n-2; j >= 0; --j)
      {
	if ( 1 == obstacleGrid[m-1][j] )
	  tbl[m-1][j] = 0;
	else
	  tbl[m-1][j] = tbl[m-1][j+1];
      }
    for (int i = m-2; i >= 0; --i)
      for (int j = n-2; j >= 0; --j)
	{
	  if ( 1 == obstacleGrid[i][j] )
	    tbl[i][j] = 0;
	  else
	    tbl[i][j] = tbl[i+1][j] + tbl[i][j+1];
	}

    return tbl[0][0];
  }
};

int main()
{
  Solution sol;
  
}

/**
 * LeetCode Problem 64
 *
 * Minimum path sum
 */

#include <vector>
#include <iostream>

using namespace std;

class Solution {
public:
  int minPathSum(vector<vector<int> > &grid)
  {
    int m = grid.size();
    if ( 0 == m ) return -1;
    int n = grid[0].size();
    if ( 0 == n ) return -1;

    vector< vector<int> > tbl(m, vector<int>(n, 0));
    tbl[m-1][n-1] = grid[m-1][n-1];

    if ( m > 1 )
      for (int i = m-2; i >= 0; --i) // right boundary
	tbl[i][n-1] = tbl[i+1][n-1] + grid[i][n-1];
    if ( 1 == n ) return tbl[0][n-1];

    for (int j = n-2; j >= 0; --j) // bottom boundary
      tbl[m-1][j] = tbl[m-1][j+1] + grid[m-1][j];
    if ( 1 == m ) return tbl[m-1][0];

    for (int i = m-2; i >= 0; --i)
      for (int j = n-2; j >= 0; --j)
	{
	  int dn = tbl[i+1][j];
	  int rt = tbl[i][j+1];
	  tbl[i][j] = min(dn, rt) + grid[i][j];
	}

    return tbl[0][0];
  }
};

int main()
{
  const int m = 3;
  const int n = 4;
  int A[m][n] = {{1,2,3,1},
		 {2,5,4,7},
		 {3,0,3,5}};
  vector< vector<int> > grid;
  for (int i = 0; i < m; ++i)
    grid.push_back(vector<int>(A[i], A[i] + n));

  Solution sol;
  cout << sol.minPathSum(grid) << endl;
}

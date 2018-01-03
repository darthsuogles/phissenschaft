/**
 * LeetCode Problem 200
 *
 * Find number of islands
 */

#include <vector>
#include <queue>
#include <iostream>
#include <utility>

using namespace std;

class Solution {
public:
  int numIslands(vector< vector<char> > &grid)
  {
    if ( grid.empty() ) return 0;
    int m = grid.size();
    int n = grid[0].size();
    if ( 0 == n ) return 0;
        
    int cnt = 0;
    //queue< pair<int, int> > vert_queue;
    vector< vector<bool> > is_visited(m, vector<bool>(n, false));
        
    for (int i = 0; i < m; ++i)
      {
	for (int j = 0; j < n; ++j)
	  {
	    if ( '1' != grid[i][j] || is_visited[i][j] ) continue;

	    ++cnt;
	    queue< pair<int, int> > vert_queue;
	    vert_queue.push(pair<int, int>(i, j));
	    
	    while ( ! vert_queue.empty() )
	      {
		auto vert = vert_queue.front(); vert_queue.pop();
		int u = vert.first, v = vert.second;
		if ( is_visited[u][v] )
		  continue;
		is_visited[u][v] = true;
                
		if ( u+1 < m && '1' == grid[u+1][v] )
		  vert_queue.push(pair<int, int>(u+1, v));
		if ( u > 0  && '1' == grid[u-1][v] )
		  vert_queue.push(pair<int, int>(u-1, v));
		if ( v+1 < n && '1' == grid[u][v+1] )
		  vert_queue.push(pair<int, int>(u, v+1));
		if ( v > 0 && '1' == grid[u][v-1] )
		  vert_queue.push(pair<int, int>(u, v-1));            
	      }
	  }
      }

    return cnt;
  }
};

int main()
{
  Solution sol;
  const char *A[] = {"11000",
		     "11000",
		     "00100",
		     "00011"};
  int m = sizeof(A) / sizeof(A[0]);
  int n = sizeof(A[0]) / sizeof(char);
  vector< vector<char> > grid;
  for (int i = 0; i < m; ++i)
    grid.push_back(vector<char>(A[i], A[i]+n));

  cout << sol.numIslands(grid) << endl;
}

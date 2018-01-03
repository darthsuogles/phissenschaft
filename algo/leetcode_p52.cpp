/**
 * LeetCode Problem 52
 *
 * N queens counts
 */

#include <vector>
#include <string>
#include <iostream>

using namespace std;

class Solution {
  bool place_queens(vector<int> &pos, int idx, int n, int &res)
  {
    if ( idx > n ) return false;
    if ( idx == n )
      {
	if ( pos[n-1] == -1 ) return false;
	// Paint the board
	++res;
	return true;
      }

    vector<bool> pos_avail(n, true);
    // Compute for horizontal attack
    for (int i = 0; i < n; ++i)
      {
	int j = pos[i];
	if ( -1 == j ) continue;

	pos_avail[j] = false;
	int d = idx - i;
	if ( j >= d )
	  pos_avail[j-d] = false;
	if ( j + d < n )
	  pos_avail[j+d] = false;
      }
    
    for (int k = 0; k < n; ++k)
      {
	if ( ! pos_avail[k] ) continue;
	
	pos[idx] = k;
	place_queens(pos, idx+1, n, res);
	pos[idx] = -1;
      }
    return false;
  }
  
public:
  int totalNQueens(int n)
  {
    int res = 0;
    if ( 0 == n ) return res;
    
    vector<int> pos(n, -1);
    place_queens(pos, 0, n, res);
    return res;
  }
};

int main()
{
  Solution sol;
  cout << sol.totalNQueens(1) << endl;
  cout << sol.totalNQueens(2) << endl;
  cout << sol.totalNQueens(3) << endl;
  cout << sol.totalNQueens(4) << endl;
}

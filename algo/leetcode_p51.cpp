/**
 * LeetCode Problem 51
 *
 * N queens
 */

#include <vector>
#include <string>
#include <iostream>

using namespace std;

class Solution {
  bool place_queens(vector<int> &pos, int idx, int n, vector< vector<string> > &res)
  {
    if ( idx > n ) return false;
    if ( idx == n )
      {
	if ( pos[n-1] == -1 ) return false;
	// Paint the board
	vector<string> board(n, string(n, '.'));
	for (int i = 0; i < n; ++i)
	  board[pos[i]][i] = 'Q';
	res.push_back(board);
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
  vector< vector<string> > solveNQueens(int n)
  {
    vector< vector<string> > res;
    if ( 0 == n ) return res;
    
    vector<int> pos(n, -1);
    place_queens(pos, 0, n, res);
    return res;
  }
};

int main()
{
  Solution sol;
  vector< vector<string> > res = sol.solveNQueens(8);
  for (auto it = res.begin(); it != res.end(); ++it)
    {
      for (auto jt = it->begin(); jt != it->end(); ++jt)
	cout << *jt << endl;
      cout << endl;
    }
}

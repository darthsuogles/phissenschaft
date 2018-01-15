/**
 * LeetCode Problem 130
 *
 * Surrounded region
 */

#include <vector>
#include <iostream>
#include <queue>
#include <utility>
//#include "test_p130.hpp"

using namespace std;

class Solution {    
  int m;
  int n;
    
  void flip_board(int i, int j, vector< vector<char> > &board)
  {
    if ( board[i][j] == 'F' ) return;

    queue< pair<int, int> > brd_q;
    brd_q.push( pair<int, int>(i, j) );
    board[i][j] = 'F';
    
    while ( ! brd_q.empty() )
      {
	auto coord = brd_q.front(); brd_q.pop();	
	i = coord.first;
	j = coord.second;	
	
	if ( i > 0 && board[i-1][j] == 'O' )
	  {
	    board[i-1][j] = 'F';
	    brd_q.push( pair<int, int>(i-1, j) );
	  }
	if ( i+1 < m && board[i+1][j] == 'O' )
	  {
	    board[i+1][j] = 'F';
	    brd_q.push( pair<int, int>(i+1, j) );
	  }
	if ( j > 0 && board[i][j-1] == 'O' )
	  {
	    board[i][j-1] = 'F';
	    brd_q.push( pair<int, int>(i, j-1) );
	  }
	if ( j+1 < n && board[i][j+1] == 'O' )
	  {
	    board[i][j+1] = 'F';
	    brd_q.push( pair<int, int>(i, j+1) );
	  }
      }
  }
    
public:
  void solve(vector<vector<char> > &board)
  {
    if ( board.empty() ) return;
    this->m = board.size();
    if ( board[0].empty() ) return;
    this->n = board[0].size();
        
    for (int i = 0; i < m; ++i)
      {
	if ( board[i][0] == 'O' )
	  flip_board(i, 0, board);
	if ( board[i][n-1] == 'O' )
	  flip_board(i, n-1, board);
      }
    for (int j = 0; j < n; ++j)
      {
	if ( board[0][j] == 'O' )
	  flip_board(0, j, board);
	if ( board[m-1][j] == 'O' )
	  flip_board(m-1, j, board);
      }
        
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
	if ( board[i][j] == 'O' )
	  board[i][j] = 'X';

    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
	if ( board[i][j] == 'F' )
	  board[i][j] = 'O';
  }
};

void print_board(vector< vector<char> > &board)
{
  for (auto it = board.begin(); it != board.end(); ++it)
    {
      for (auto jt = it->begin(); jt != it->end(); ++jt)
	cout << *jt << " ";
      cout << endl;
    }
}

int main()
{
  Solution sol;

  const char* A[] = {
    "XXXX",
    "XOOX",
    "XXOX",
    "XOXX",
  };
  
  vector< vector<char> > board;
  int m = sizeof(A) / sizeof(A[0]);
  int n = strlen(A[0]);
  cout << m << " " << n << endl;
  for (int i = 0; i < m; ++i)
    board.push_back( vector<char>(A[i], A[i] + n) );

  print_board(board);
  cout << "----------------------" << endl;
  sol.solve(board);
  print_board(board);
}

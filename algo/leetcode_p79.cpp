/**
 * LeetCode Problem 79
 *
 * Word search
 */

#include <iostream>
#include <string>
#include <vector>

using namespace std;

class Solution {
  bool search_word(int i, int j, vector< vector<bool> > &is_used, vector< vector<char> > &board, string word)
  {
    if ( word.empty() ) return true;
    if ( is_used[i][j] || board[i][j] != word[0] ) return false;
    if ( word.size() == 1 ) return true;
    
    is_used[i][j] = true;
    int m = board.size();
    int n = board[0].size();
    string next_word = word.substr(1, word.size() -1);
    if ( i > 0 )
      if ( search_word(i-1, j, is_used, board, next_word) )
	return true;
    if ( i < m-1 )
      if ( search_word(i+1, j, is_used, board, next_word) )
	return true;
    if ( j > 0 )
      if ( search_word(i, j-1, is_used, board, next_word) )
	return true;
    if ( j < n-1 )
      if ( search_word(i, j+1, is_used, board, next_word) )
	return true;
    
    is_used[i][j] = false;
    return false;
  }
public:
  bool exist(vector<vector<char> > &board, string word)
  {
    if ( word.empty() ) return true;
    if ( board.empty() ) return false;
    int len = word.size();
    char ch = word[0];
        
    int m = board.size();
    int n = board[0].size();
    vector< vector<bool> > is_used(m, vector<bool>(n, false));
    for ( int i = 0; i < m; ++i )
      for ( int j = 0; j < n; ++j )
	if ( board[i][j] == ch )
	  if ( search_word(i, j, is_used, board, word) )
	    return true;
    return false;
  }
};

int main()
{
  Solution sol;

  const char *A[] = {  "ABCE",
		       "SFCS",
		       "ADEE" };
  int n = sizeof(A[0]);
  int m = sizeof(A) / n;
  vector< vector<char> > board;
  for (int i = 0; i < m; ++i)
    board.push_back(vector<char>(A[i], A[i] + n));

  cout << sol.exist(board, "ABCCED") << endl;
  cout << sol.exist(board, "SEE") << endl;
  cout << sol.exist(board, "ABCB") << endl;
}

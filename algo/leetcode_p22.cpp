/**
 * LeetCode Problem 22 
 *
 * Generate all strings with n matching parentheses
 */

#include <iostream>
#include <vector>
#include <string>

using namespace std;

class Solution
{
  void grid_traversal(string prefix, int i, int j, int n, vector<string> &res)
  {
    if ( i == n && j == n )
      {
	res.push_back(prefix);
	return;
      }

    if ( i < n )
      grid_traversal(prefix + '(', i+1, j, n, res);
    if ( j < i && j < n )
      grid_traversal(prefix + ')', i, j+1, n, res);
  }
  
public:
  vector<string> generateParenthesis(int n)
  {
    vector<string> res;
    if ( 0 == n ) return res;
    grid_traversal("(", 1, 0, n, res);
    return res;
  }
};

int main()
{
  Solution sol;

  vector<string> res = sol.generateParenthesis(3);
  for ( int i = 0; i < res.size(); ++i )
    cout << res[i] << endl;
}

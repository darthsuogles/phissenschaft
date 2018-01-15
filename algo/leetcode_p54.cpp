/**
 * LeetCode Problem 54
 *
 * Spiral matrix traversal
 */

#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
  vector<int> spiralOrder(vector<vector<int> > &matrix)
  {
    vector<int> res;		
    int m = matrix.size();
    if ( 0 == m ) return res;
    int n = matrix[0].size();	
   
    int up = 0, down = m;
    int left = -1, right = n;
    int i = 0, j = 0;
    while ( left < right && up < down )
      {
	for (; j < right; ++j) res.push_back(matrix[i][j]);
	++i; --j; right -= 1;
	if ( i == down ) break;

	for (; i < down; ++i) res.push_back(matrix[i][j]);
	--i; --j; down -= 1;
	if ( j == left ) break;

	for (; j > left; --j) res.push_back(matrix[i][j]);
	--i; ++j; left += 1;
	if ( i == up ) break;

	for (; i > up; --i) res.push_back(matrix[i][j]);
	++i; ++j; up += 1;
	if ( j == right ) break;
      }

    return res;
  }
};

int main()
{
  Solution sol;
  const int m = 1;
  const int n = 2;
  int A[m][n] = {{2,3}};
  vector< vector<int> > matrix;
  for (int i = 0; i < m; ++i)
    matrix.push_back( vector<int>(A[i], A[i]+n) );

  vector<int> res = sol.spiralOrder(matrix);
  for (auto it = res.begin(); it != res.end(); ++it)
    cout << *it << " ";
  cout << endl;
}

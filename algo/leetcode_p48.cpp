/**
 * LeetCode Problem 48
 *
 * Rotate image
 */

#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
  void rotate(vector<vector<int> > &matrix)
  {
    int n = matrix.size();
    // vector< vector<int> > A(n, vector<int>(n, 0));
    // for (int i = 0; i < n; ++i)
    //   for (int j = 0; j < n; ++j)
    // 	A[j][n-1-i] = matrix[i][j];
            
    // for (int i = 0; i < n; ++i)
    //   for (int j = 0; j < n; ++j)
    // 	matrix[i][j] = A[i][j];
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < i; ++j)
	{
	  int tmp = matrix[i][j];
	  matrix[i][j] = matrix[j][i];
	  matrix[j][i] = tmp;
	}

    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n/2; ++j)
	{
	  int tmp = matrix[i][j];
	  matrix[i][j] = matrix[i][n-1-j];
	  matrix[i][n-1-j] = tmp;
	}
  }
};

int main()
{
  Solution sol;
  const int n = 2;
  int A[][n] = {{1,2}, {3,4}};
  vector< vector<int> > matrix;
  matrix.push_back( vector<int>(A[0], A[0] + n) );
  matrix.push_back( vector<int>(A[1], A[1] + n) );

  sol.rotate(matrix);
  for (int i = 0; i < n; ++i)
    {
      for (int j = 0; j < n; ++j)
	cout << matrix[i][j] << " ";
      cout << endl;
    }
}

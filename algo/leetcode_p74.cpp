/**
 * LeetCode Problem 72
 *
 * Edit distance: insert, delete, replace
 */

#include <iostream>
#include <vector>

using namespace std;

class Solution {
  bool binary_search(vector<int> &vs, int target)
  {
    int n = vs.size();
    if ( vs[0] == target || vs[n-1] == target )
      return true;

    int a = 0, b = n-1;    
    while ( a + 1 < b )
      {
	int mid = (a+b)/2;
	int val = vs[mid];
	if ( val == target )
	  return true;
	else if ( val < target )
	  a = mid;
	else
	  b = mid;
      }
    if ( vs[a] == target || vs[b] == target )
      return true;
    else
      return false;
  }
  
public:
  bool searchMatrix(vector<vector<int> > &matrix, int target)
  {
    if ( matrix.empty() ) return false;
    if ( matrix[0].empty() ) return false;
    int m = matrix.size();
    int n = matrix[0].size();

    if ( target < matrix[0][0] || target > matrix[m-1][n-1] )
      return false;
    if ( target >= matrix[m-1][0] )
      return binary_search(matrix[m-1], target);
    
    // Binary search on the initial row
    int a = 0, b = m-1;
    while ( a + 1 < b )
      {
	int mid = (a+b)/2;
	int val = matrix[mid][0];
	if ( val == target )
	  return true; // found it
	else if ( val < target )
	  a = mid;
	else // val > target
	  b = mid;
      }
    if ( matrix[a][0] == target || matrix[b][0] == target )   
      return true;

    if ( matrix[b][0] <= target )
      return binary_search(matrix[b], target);
    else
      return binary_search(matrix[a], target);
  }
};

int main()
{
  Solution sol;

#define test_case(A, target) {				\
    int n = sizeof(A[0]) / sizeof(int);			\
    int m = sizeof(A) / sizeof(A[0]);			\
    vector< vector<int> > matrix;			\
    for (int i = 0; i < m; ++i)				\
      matrix.push_back(vector<int>(A[i], A[i] + n));	\
    cout << sol.searchMatrix(matrix, target) << endl; }	\

  int A[][4] = {
    {1,   3,  5,  7},
    {10, 11, 16, 20},
    {23, 30, 34, 50}
  };
  test_case(A, 3);
  test_case(A, 5);
  test_case(A, 19);
  test_case(A, 24);
  test_case(A, 23);

  int B[][1] = {{1}, {3}};
  test_case(B, 1);
  test_case(B, 3);
}

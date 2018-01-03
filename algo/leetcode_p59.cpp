/**
 * LeetCode Problem 59
 *
 * Spiral matrix for 1 to n^2
 */

#include <vector>
#include <iostream>

using namespace std;

class Solution {
public:
  vector<vector<int> > generateMatrix(int n)
  {
    if ( 0 == n )
      return vector< vector<int> >(0);

    vector< vector<int> > res(n, vector<int>(n, 0));
    int n2 = n * n;

    int north = 0;
    int south = n;
    int west = -1;	
    int east = n;

    int cnt = 1;
    int i = 0, j = 0;
    while ( north < south && west < east )
      {
	for (; j < east; ++j ) // ->
	  res[i][j] = cnt++;
	++i; --j; --east;
	if ( east == west ) break;

	for (; i < south; ++i ) // downward
	  res[i][j] = cnt++;
	--i; --j; --south;
	if ( south == north ) break;

	for (; j > west; --j ) // <-
	  res[i][j] = cnt++;
	--i; ++j; ++west;
	if ( east == west ) break;

	for (; i > north; --i ) // upward
	  res[i][j] = cnt++;
	++i; ++j; ++north;
	if ( south == north ) break;
      }

    return res;
  }
};

int main()
{
  Solution sol;
#define test_case(n)  {							\
    vector< vector<int> > res = sol.generateMatrix((n));		\
    for (auto it = res.begin(); it != res.end(); ++it)			\
      {									\
	for (auto jt = it->begin(); jt != it->end(); ++jt)		\
	  cout << *jt << "\t";						\
	cout << endl;							\
      }									\
    cout << "--------------" << endl;					\
  }									\

  for (int i = 1; i < 6; ++i)
    test_case(i);
}

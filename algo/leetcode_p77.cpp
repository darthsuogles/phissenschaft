/**
 * LeetCode Problem 77
 *
 * Combinations
 *
 * Given two integers n and k, return all possible combinations of k numbers out of 1 ... n.
 */

#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
  vector<vector<int> > combine(int n, int k)
  {
    vector< vector<int> > res;
    if ( 0 == n || 0 == k ) return res;
    if ( 1 == k )
      {
	for (int i = 1; i <= n; ++i)
	  res.push_back(vector<int>(1, i));
	return res;
      }
    if ( n == k )
      {
	vector<int> curr(n, 0);
	for (int i = 0; i < n; ++i)
	  curr[i] = i+1;
	res.push_back(curr);
	return res;
      }
    
    res = combine(n-1, k);
    vector< vector<int> > res_sans_n = combine(n-1, k-1);
    for (auto it = res_sans_n.begin(); it != res_sans_n.end(); ++it)
      {
	it->push_back(n);
	res.push_back(*it);
      }

    return res;
  }
};

int main()
{
  Solution sol;

  int n = 4;
  int k = 2;
  vector< vector<int> > res = sol.combine(n, k);
  for (auto it = res.begin(); it != res.end(); ++it)
    {
      for (auto jt = it->begin(); jt != it->end(); ++jt)
	cout << *jt << " ";
      cout << endl;
    }
}

/**
 * LeetCode Problem 47
 *
 * Permutation with duplicates
 */

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
  vector< vector<int> > interleave(vector<int> &lst, int bnd, int a, int cnt)
  {
    vector< vector<int> > res;
    if ( 0 == bnd )
      {
	if ( 0 == cnt ) return res;
	res.push_back( vector<int>(cnt, a) );
	return res;
      }
    if ( 0 == cnt )
      {
	res.push_back( vector<int>(lst.begin(), lst.begin() + bnd) );
	return res;
      }
    
    vector< vector<int> > r1 = interleave(lst, bnd, a, cnt-1);
    for (auto it = r1.begin(); it != r1.end(); ++it)
      {
	it->push_back(a);
	res.push_back(*it);
      }
    vector< vector<int> > r2 = interleave(lst, bnd-1, a, cnt);
    for (auto it = r2.begin(); it != r2.end(); ++it)
      {
	it->push_back(lst[bnd-1]);
	res.push_back(*it);
      }
    return res;
  }
  
  vector< vector<int> > perm_uniq(vector<int> &num, int bnd)
  {
    vector< vector<int> > res;
    if ( 0 == bnd ) return res;
    if ( 1 == bnd )
      {
	res.push_back(vector<int>(1, num[0]));
	return res;
      }
    
    int a = num[bnd-1];
    int idx = bnd-1;
    for (; idx >= 0; --idx)
      if ( num[idx] != a ) break;
    if ( idx < 0 )
      {
	res.push_back( vector<int>(bnd, a) );
	return res;
      }
    ++idx;
    vector< vector<int> > hs = perm_uniq(num, idx);

    // Now interleave the elements of the two arrays
    for ( auto it = hs.begin(); it != hs.end(); ++it )
      {
	vector< vector<int> > curr = interleave(*it, it->size(), a, bnd-idx);
	for ( auto jt = curr.begin(); jt != curr.end(); ++jt )
	  res.push_back(*jt);
      }

    return res;
  }
  
public:
  vector<vector<int> > permuteUnique(vector<int> &num)
  {
    sort(num.begin(), num.end());
    return perm_uniq(num, num.size());
  }
};

int main()
{
  Solution sol;
  int A[] = {1,1,2};
  int len = sizeof(A) / sizeof(int);
  vector<int> num(A, A+len);
  vector< vector<int> > res = sol.permuteUnique(num);
  for (auto it = res.begin(); it != res.end(); ++it)
    {
      for (auto jt = it->begin(); jt != it->end(); ++jt)
	cout << *jt << " ";
      cout << endl;
    }
}

/**
 * LeetCode Problem 90
 *
 * Subsets with duplicate terms
 */

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
  vector< vector<int> > find_subsets(vector<int>::iterator init, vector<int>::iterator fini)
  {
    vector< vector<int> > res;
    if ( init == fini )
      {
	res.push_back(vector<int>(0));
	return res;
      }
    
    int cnt = 1;
    for (auto it = fini-1; it > init; --it, ++cnt)
      if ( *it != *(it-1) ) break;

    int elem = *(fini-1);	
    vector< vector<int> > res_tail = find_subsets(init, fini-cnt);
    res.insert(res.end(), res_tail.begin(), res_tail.end());
    for (int k = 1; k <= cnt; ++k)
      {		
	for (auto it = res_tail.begin(); it != res_tail.end(); ++it)
	  {
	    it->push_back(elem);
	    res.push_back(*it);
	  }
      }

    return res;
  }
  
public:
  vector<vector<int> > subsetsWithDup(vector<int> &S)
  {
    vector< vector<int> > res;
    if ( S.empty() )
      {
	res.push_back(vector<int>(0));
	return res;
      }
    sort(S.begin(), S.end());

    return find_subsets(S.begin(), S.end());
  }
};


int main()
{
  Solution sol;
  int A[] = {1,1};
  int len = sizeof(A) / sizeof(int);
  vector<int> seq(A, A+len);

  auto res = sol.subsetsWithDup(seq);
  for (auto it = res.begin(); it != res.end(); ++it)
    {
      cout << "[ ";
      for (auto jt = it->begin(); jt != it->end(); ++jt)
	cout << *jt << " ";
      cout << "]" << endl;
    }
}

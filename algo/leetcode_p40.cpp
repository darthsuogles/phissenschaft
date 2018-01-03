/**
 * LeetCode Problem 40
 *
 * Combination sum II
 */

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
  vector< vector<int> > comb_sum(vector<int>::iterator init,
				 vector<int>::iterator fini,
				 int target)
  {
    vector< vector<int> > res;
    if ( init >= fini ) return res;
    if ( *init > target ) return res;

    // Explicitly count the duplicates
    int a = *(fini - 1);
    int cnt = 0;
    for (auto it = init; it != fini; ++it)
      if ( *it == a ) ++cnt;
    
    for (int i = 0, val = 0; i <= cnt; ++i, val += a)
      {
	if ( val == target )
	  res.push_back(vector<int>(i, a));
	
	vector< vector<int> > res_avec = comb_sum(init, fini - cnt, target - val);	
	for (auto it = res_avec.begin(); it != res_avec.end(); ++it)
	  {
	    for (int j = 0; j < i; ++j)
	      it->push_back(a);	      
	    res.push_back(*it);
	  }
      }
    
    return res;
  }
  
public:
  vector<vector<int> > combinationSum2(vector<int> &num, int target)
  {
    sort(num.begin(), num.end());
    return comb_sum(num.begin(), num.end(), target);
  }
};

int main()
{
  Solution sol;
  
#define test_case(A, target)  {						\
    vector<int> num(A, A + sizeof(A) / sizeof(int));			\
    vector< vector<int> > res = sol.combinationSum2(num, target);	\
    for (auto it = res.begin(); it != res.end(); ++it)			\
      {									\
	for (auto jt = it->begin(); jt != it->end(); ++jt)		\
	  cout << *jt << " ";						\
	cout << endl;							\
      }									\
  }									\

  int A[] = {10,1,2,7,6,1,5};
  test_case(A, 8);
  
  int B[] = {2,2,2};
  test_case(B, 4);
}

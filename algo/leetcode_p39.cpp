/**
 * LeetCode Problem 39
 * 
 * Combination sum
 */

#include <iostream>
#include <vector>

using namespace std;

class Solution {
  vector< vector<int> > comb_sum(vector<int>::iterator init,
				 vector<int>::iterator fini,
				 int target)
  {    
    vector< vector<int> > res;
    if ( init >= fini ) return res;
    if ( target < *init ) return res;
    
    int a = *(fini - 1);
    int cnt, val;
    for (cnt = 0, val = 0; val < target; ++cnt, val += a)
      {
	vector< vector<int> > subres = comb_sum(init, fini-1, target - val);
	for (int j = 0; j < subres.size(); ++j)
	  {
	    for (int k = 0; k < cnt; ++k)
	      subres[j].push_back(a);
	    res.push_back( subres[j] );
	  }
      }

    if ( val == target )
      {
	vector<int> curr;
	for (int k = 0; k < cnt; ++k)
	  curr.push_back(a);
	res.push_back(curr);
      }

    return res;
  }
  
public:
  vector< vector<int> > combinationSum(vector<int> &candidates, int target)
  {
    vector< vector<int> > res;
    
    auto init = candidates.begin();
    auto fini = candidates.end();
    if ( init == fini ) return res;

    sort(candidates.begin(), candidates.end());
    return comb_sum(init, fini, target);
  }
};

int main()
{
  Solution sol;
  int A[] = {8,7,4,3};
  int target = 11;
  vector<int> candidates(A, A + sizeof(A) / sizeof(int));
  vector< vector<int> > res = sol.combinationSum(candidates, target);
  for (int i = 0; i < res.size(); ++i)
    {
      for (int j = 0; j < res[i].size(); ++j)
	cout << res[i][j] << " ";
      cout << endl;
    }
}

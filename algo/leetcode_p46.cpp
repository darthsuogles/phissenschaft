/**
 * LeetCode Problem 46
 *
 * All permutations
 */

#include <iostream>
#include <vector>

using namespace std;

class Solution {
  vector< vector<int> > all_permutations(vector<int> &num, int bnd)
  {
    vector< vector<int> > res;
    if ( bnd == 0 ) return res;
    if ( bnd == 1 )
      {
	res.push_back( vector<int>(1, num[0]) );
	return res;
      }

    int tmp;
    for (int i = bnd-1; i >=0; --i)
      {
	int j = bnd-1;
	tmp = num[j]; num[j] = num[i]; num[i] = tmp;
	vector< vector<int> > curr = all_permutations(num, bnd-1);
	tmp = num[j]; num[j] = num[i]; num[i] = tmp;
	
	for (auto it = curr.begin(); it != curr.end(); ++it)
	  {
	    it->push_back(num[i]);
	    res.push_back(*it);
	  }
      }

    return res;

  }
  
public:
  vector<vector<int> > permute(vector<int> &num)
  {
    vector< vector<int> > res;
    int n = num.size();
    if ( 0 == n ) return res;

    return all_permutations(num, n);
  }
};

int main()
{
  Solution sol;
  int A[] = {1,2,3,4};
  int len = sizeof(A) / sizeof(int);
  vector<int> num(A, A+len);

  vector< vector<int> > res = sol.permute(num);
  for (auto it = res.begin(); it != res.end(); ++it)
    {
      for (int i = 0; i < it->size(); ++i)
	cout << (*it)[i] << " ";
      cout << endl;
    }
}

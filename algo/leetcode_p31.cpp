/**
 * LeetCode Problem 31
 *
 * Next permutation
 */

#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include <algorithm>

using namespace std;


class Solution {
public:
  void nextPermutation(vector<int> &num)
  {
    if ( num.size() <= 1 ) return;
    auto init = num.begin();
    auto fini = num.end();

    for (; init < fini - 1; ++init)
      {
	if ( is_sorted(init + 1, fini, std::greater<int>()) ) // reversely sorted
	  {
	    if ( *init >= *(init + 1) ) // whole array reversely sorted
	      reverse(init, fini);
	    else
	      {	    
		reverse(init + 1, fini);
		auto upper = upper_bound(init + 1, fini, *init);
		int tmp = *init;
		*init = *upper;
		*upper = tmp;
	      }
	    return;
	  }
      }
  }
};

int main()
{
  Solution sol;
  int A[] = {1,2,3,1};
  vector<int> num(A, A + sizeof(A) / sizeof(int));
  sol.nextPermutation(num);
  for (int i = 0; i < num.size(); ++i)
    cout << num[i] << " ";
  cout << endl;
}

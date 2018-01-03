/**
 * LeetCode Problem 72
 *
 * Edit distance: insert, delete, replace
 */

#include <iostream>
#include <vector>
#include <limits.h>

using namespace std;

class Solution {
public:
  int minDistance(string word1, string word2)
  {
    if ( word1 == word2 )
      return 0;
    if ( word1.empty() )
      return word2.size();
    if ( word2.empty() )
      return word1.size();

    // Perform alignment: longest common subsequence
    int m = word1.size();
    int n = word2.size();
    vector<int> tbl(n);
    for (int j = 0; j < n; ++j) tbl[j] = j+1;
    for (int i = 0; i < m; ++i)
      {
	int vd = i; // word[0:(i-1)] to empty
	int vl = i+1; // word1[0:i] to empty
	int vu;
	char ch = word1[i];
	for (int j = 0; j < n; ++j)
	  {
	    vu = tbl[j];
	    vl = tbl[j] = min(min(vl, vu) + 1,
			      vd + (ch != word2[j]));
	    vd = vu;
	  }	
      }

    return tbl[n-1];
  }
};

int main()
{
  Solution sol;

  cout << sol.minDistance("abc", "abcd") << endl;    
  cout << sol.minDistance("obama", "osama") << endl;
  cout << sol.minDistance("abd", "abcd") << endl;
}

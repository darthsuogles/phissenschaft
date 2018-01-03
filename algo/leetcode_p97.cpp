/**
 * Intervleaving strings
 *
 * Given s1, s2, s3, find whether s3 is formed by the interleaving of s1 and s2.
 * https://leetcode.com/problems/interleaving-string/
 */

#include <iostream>
#include <unordered_map>

using namespace std;

class Solution {
  unordered_map<int, unordered_map<int, bool> > tbl;

  int partialRes(int m, int n) {
    if ( 0 == tbl.count(m) ) return -1;
    if ( 0 == tbl[m].count(n) ) return -1;
    return tbl[m][n] ? 1 : 0;
  }
  
public:
  bool isInterleave(string s1, string s2, string s3, bool isInit=true) {
    if ( s1.empty() )
      return s2 == s3;
    if ( s2.empty() )
      return s1 == s3;
    if ( s3.empty() )
      return false;

    if ( isInit ) tbl.clear();
    
    int m = s1.size();
    int n = s2.size();
    int pres = partialRes(m, n);
    if ( -1 != pres )
      return 1 == pres;
    
    char ch1 = s1[0];
    char ch2 = s2[0];
    char ch3 = s3[0];
    if ( ch1 == ch3 ) {
      if ( isInterleave(s1.substr(1), s2, s3.substr(1), false) )
	return tbl[m][n] = true;
    }
    if ( ch2 == ch3 )
      return tbl[m][n] = isInterleave(s1, s2.substr(1), s3.substr(1), false);
    return tbl[m][n] = false;
  }
};

int main() {
  Solution sol;
  cout << boolalpha;
  cout << sol.isInterleave("aabcc", "dbbca", "aadbbcbcac") << endl;
  cout << sol.isInterleave("aabcc", "dbbca", "aadbbbaccc") << endl;
}

/**
 * LeetCode Problem 87
 *
 * Scrambled strings
 */

#include <iostream>
#include <string>

using namespace std;

class Solution {
public:
  bool isScramble(string s1, string s2)
  {
    if ( s1.empty() ) return s2.empty();
    if ( s2.empty() ) return false;
    if ( s1 == s2 ) return true;
    if ( s1.size() != s2.size() ) return false;
        
    int n = s1.size();    
    const int nbins = 128;
    int tbl[nbins] = {0};
    for (int i = 0; i < n; ++i)
      {
	++tbl[ s1[i] ];
	--tbl[ s2[i] ];
      }
    for (int c = 0; c < nbins; ++c)
      if ( tbl[c] != 0 ) return false;
            
    // Find the length-k prefix 0 ... k-1
    int tbl_bk[nbins] = {0}; // counting from the back
    for (int k = 1; k < n; ++k)
      {
	++tbl[ s1[k-1] ];
	--tbl[ s2[k-1] ];
	bool is_match = true;
	for (int c = 0; c < nbins; ++c)
	  if ( tbl[c] != 0 ) { is_match = false; break; }
	
	if ( is_match ) // could be a scramble site
	  {
	    // If both substrings are scrambled
	    if ( isScramble(s1.substr(0, k), s2.substr(0, k)) )
	      if ( isScramble(s1.substr(k, n-k), s2.substr(k, n-k)) )
		return true;
	  }

	// Looking from the back
	++tbl_bk[ s1[n-k] ];
	--tbl_bk[ s2[k-1] ];
	is_match = true;
	for ( int c = 0; c < nbins; ++c )
	  if ( tbl_bk[c] != 0  ) { is_match = false; break; }

	if ( is_match )
	  {
	    if ( isScramble(s1.substr(n-k, k), s2.substr(0, k)) )
	      if ( isScramble(s1.substr(0, n-k), s2.substr(k, n-k)) )
		return true;		  
	  }
      }
    return false;
  }
};

int main()
{
  Solution sol;

  cout << sol.isScramble("rgeat", "great") << endl;
}

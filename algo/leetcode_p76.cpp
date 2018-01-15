/**
 * LeetCode Problem 76
 *
 * Minimum window substring
 */

#include <iostream>
#include <string>

using namespace std;

class Solution {

#pragma inline
  bool is_substr_match(int curr[], int patt[])
  {
    for (int d = 0; d < 256; ++d)
      if ( curr[d] < patt[d] )
	return false;
    return true;
  }
  
public:
  string minWindow(string S, string T)
  {
    if ( S.empty() || T.empty() ) return "";
    if ( S.size() < T.size() ) return "";
    string res;
    int len = S.size();

    // Construct a table
    int patt[256] = {0};
    for (int k = 0; k < T.size(); ++k)
      ++patt[ T[k] ];

    int min_window_size = len;
    int min_init = 0;
    int curr[256] = {0};
    int i = 0, j = 0; // consider the substring from j to i
    bool is_ever_matched = false;
    for (; i < len; ++i )
      {
	++curr[ (int)S[i] ];

	// Compare current histogram with the pattern
	if ( is_substr_match(curr, patt) )
	  {
	    is_ever_matched = true;
	    for (; j < i; ++j)
	      {
		int ch = S[j];
		if ( curr[ch] > 0 )
		  {
		    --curr[ch];
		    if ( ! is_substr_match(curr, patt) )
		      break;
		  }
	      } 

	    int window_size = i - j + 1;
	    if ( window_size < min_window_size )
	      {
		min_window_size = window_size;
		min_init = j;
	      }
	    ++j;
	  }
      }

    if ( ! is_ever_matched )
      return "";
    return S.substr(min_init, min_window_size);
  }
};

int main()
{
  Solution sol;
#define test_case(S, T) cout << sol.minWindow((S), (T)) << endl
  
  test_case("ADOBECODEBANC", "ABC");
  test_case("abc", "a");
  test_case("a", "aa");
  test_case("a", "b");
}

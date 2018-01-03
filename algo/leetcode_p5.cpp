/**
 * LeetCode problem 5: longest palindromic substring
 *
 * Given a string S, find the longest palindromic substring in S. 
 * You may assume that the maximum length of S is 1000, and 
 * there exists one unique longest palindromic substring
 */

#include <iostream>
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
  string longestPalindrome(string s) {
    int len = s.size();
    string parlin_str = "";

    // Substrings with odd length
    for (int i=0; i < len; ++i)
      {
	int bnd = min(i-0, (len-1)-i);
	int k;
	for (k=0; k < bnd; ++k)
	  if ( s[i-k] != s[i+k] )
	    {
	      --k;
	      break;
	    }
	if ( k > 0 && s[i-k] != s[i+k] ) --k; // if bnd is reached
	
	if ( (2*k+1) > parlin_str.size() )
	  parlin_str = s.substr(i-k, 2*k+1);
      }

    // Substrings with even length
    for (int i=1; i < len; ++i)
      {
	int bnd = min(i-0, (len-1)-(i-1));
	int k;
	for (k=1; k < bnd; ++k)
	  if ( s[i-k] != s[(i-1)+k] )
	    {
	      --k;
	      break;
	    }
	if ( k > 0 && s[i-k] != s[(i-1)+k]) --k; // if bnd is reached
	
	if ( 2*k > parlin_str.size() )
	  parlin_str = s.substr(i-k, 2*k);
      }

    return parlin_str;
  }
};

int main()
{
  Solution sol;
#define test_case(S) cout << sol.longestPalindrome((S)) << endl
  
  test_case("abb");
  test_case("abba");
  test_case("ddcbababcae");
  
  return 0;
}

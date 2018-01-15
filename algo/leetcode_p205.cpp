/**
 * LeetCode Problem 205
 *
 * String isomorphism
 */

#include <string>
#include <vector>
#include <cassert>
#include <iostream>

using namespace std;

class Solution {
public:
  bool isIsomorphic(string s, string t)
  {    
    if ( s.empty() ) return true;
        
    // Fix t as template, Procruste
    int len = s.size();
    bool is_char_visited[256] = {false};
    vector<bool> is_replaced(len, false);
    for (int i = 0; i < len; ++i)
      {
	char ch = s[i];
	if ( is_replaced[i] || ch == t[i] )
	  {
	    is_char_visited[ ch ] = true;
	    continue;
	  }
            
	// If we have seen this char but not replaced this
	char rep = t[i];
	if ( is_char_visited[ ch ] )
	  return false;
	is_char_visited[ ch ] = true;
	for (int j = 0; j < len; ++j)
	  {
	    if ( s[j] == ch ) 
	      {
		if ( t[j] != rep || is_replaced[j] ) return false;
		is_replaced[j] = true;
	      }
	    if ( t[j] == rep )
	      if ( s[j] != ch ) return false;
	  }
      }

    return true;
  }   
};

int main()
{
  Solution sol;

#define test_case(S, T, expected) {   \
    string s = (S), t = (T); \
    bool res = sol.isIsomorphic(s, t);		\
    cout << s << " --> " << t << endl; assert( res == (expected) ); }
  
  test_case("aa", "ab", false);
  test_case("aa", "aa", true);
  test_case("egg", "add", true);
  test_case("foo", "bar", false);
  test_case("paper", "title", true);
}

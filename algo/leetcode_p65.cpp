/**
 * LeetCode Problem 64
 *
 * Valid number
 */

#include <iostream>
#include <string>
#include <cassert>

using namespace std;

class Solution {
public:
  bool isNumber(string s)
  {
    if ( s.empty() ) return false;
    int len = s.size();
    int idx = 0;
    char ch;

    bool is_with_sign = false;
    bool is_float = false;

    for (; idx < len; ++idx) // skip the initial spaces
      if ( s[idx] != ' ' ) break; if ( len == idx ) return false;

    ch = s[idx];
    if ( '+' == ch || '-' == ch ) // skip the + - sign
      ++idx; if ( len == idx ) return false;

    ch = s[idx];
    if ( '.' == ch ) // a float <= 0, skip the sign
      {
	is_float = true;
	++idx; if ( len == idx ) return false;
      }
    
    ch = s[idx]; // this guy must be a digit
    if ( ch < '0' || ch > '9' )
      return false;

    bool is_sci = false;
    int num_cnt = 0;	    
    for (; idx < len; ++idx)
      {
	char ch = s[idx];
	if ( '0' <= ch && ch <= '9' )
	  continue;
	if ( 'e' == ch ) // scientific notation
	  {
	    if ( is_sci ) // at most one 'e'
	      return false;

	    is_sci = true;
	    // Look ahead
	    if ( idx + 1 == len ) return false;
	    if ( '+' == s[idx+1] || '-' == s[idx+1] )
	      ++idx; if ( idx + 1 == len ) return false;
	    if ( s[idx+1] < '0' || s[idx+1] > '9' )
	      return false;
	    continue;
	  }
	if ( '.' == ch )
	  {
	    if ( is_sci ) // 'e' cannot preceed '.'
	      return false;
	    if ( is_float ) // '.' can only appear once
	      return false;

	    is_float = true;
	    // Look ahead
	    if ( idx + 1 == len ) return true;
	    if ( s[idx+1] == ' ' ) // skip all the spaces
	      {
		for (++idx; idx < len; ++idx)
		  if ( s[idx] != ' ' ) break;
		return ( idx == len );
	      }
	    if ( 'e' == s[idx+1] )
	      continue;
	    if ( s[idx+1] < '0' || s[idx+1] > '9' )
	      return false;
	    continue;
	  }
	if ( ' ' == ch ) 
	  {
	    for (; idx < len; ++idx)
	      if ( s[idx] != ' ' ) break;
	    return ( idx == len );
	  }
	else
	  return false;
      }
    return true;

  }
};

int main()
{
  Solution sol;
#define test_case(STR, VALID) assert( sol.isNumber((STR)) == VALID )

  test_case("0", true);
  test_case(" 0.1 ", true);
  test_case("2.", true);
  test_case("2e10", true);
  test_case("53K", false);
  test_case("+.8", true);
  test_case("+ 1", false);
  test_case(".1.", false);
  test_case("46.e3", true);
  test_case(" 005047e+6", true);
  test_case("4e+", false);
}

/**
 * LeetCode Problem 10: simple regular expression match
 *
 * '.' Matches any single character.
 * '*' Matches zero or more of the preceding element.
 *
 * The matching should cover the entire input string (not partial).
 */

#include <iostream>
#include <cstring>
#include <cassert>

using namespace std;

class Solution {
public:
  bool isMatch(const char *s, const char *p)
  {
    if ( p == NULL || s == NULL )
      return false;
    if ( *p == '\0' )
      return ( *s == '\0' );
    
    if ( p[0] == '*' ) // the first one cannot be a Kleen star
      return false; 

    if ( p[1] == '*' )
      {
	bool is_match = isMatch(s, p + 2); // match a* to empty
	if ( is_match )
	  return true;
	if ( s[0] == '\0' )
	  return is_match;

	if ( s[0] == p[0] || p[0] == '.' ) // match one character
	  return isMatch(s + 1, p);
	else
	  return false;
      }
    else
      {
	if ( p[0] == '.' || p[0] == s[0] )
	  return isMatch(s + 1, p + 1);
	else // s[0] != p[0]
	  return false;
      }      
  }

  void test_case(const char *s, const char *p, bool is_match)
  {
    cout << s << "\t:\t" << p << endl;
    assert(isMatch(s, p) == is_match);
  }
};

int main()
{
  Solution sol;
  sol.test_case("aa", "a", false);
  sol.test_case("aa", "aa", true);
  sol.test_case("aaa", "a*", true);
  sol.test_case("ab", ".*", true);
  sol.test_case("aaaa", "a", false);
  sol.test_case("aab", "c*a*b", true);
  sol.test_case("ab", ".*c", false);
  sol.test_case("aaa", "aaaa", false);
  sol.test_case("aaa", "a*a", true);
  sol.test_case("aaaaaaaaaaaaab", "a*a*a*a*a*a*a*a*a*a*c", false);
  
  return 0;
}

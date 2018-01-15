/**
 * LeetCode Problem 44
 *
 * Wild card matching
 */

#include <iostream>
#include <cstring>

using namespace std;

class Solution {
public:
  // Brute force solution
  bool isMatch_v0(const char *s, const char *p)
  {
    if ( NULL == s || '\0' == s[0] ) // only sequence of '*' can match empty string
      {
	if ( NULL == p || '\0' == p[0] ) return true;
	int idx = 0;
	while ( p[idx] != '\0' )
	  {
	    if ( p[idx++] != '*' )
	      return false;
	  }
	return true;
      }
    if ( NULL == p || '\0' == p[0] ) return false;
    // Check strlen
    int n = strlen(s);
    int len = 0;
    for (int i = 0; p[i] != '\0'; ++i) len += (p[i] != '*');
    if ( n < len ) return false;

    static int iter = 0;
    printf("%d:\t %s:\n\t %s\n", ++iter, s, p);

    char ch = s[0];
    char pat = p[0];
    if ( ch == pat || '?' == pat ) return isMatch(s+1, p+1);
    if ( '*' == pat ) // wild card match
      {
	int idx = 0;
	for (; p[idx] == '*' && p[idx] != '\0'; ++idx);
	if ( '\0' == p[idx] ) // the pattern '*' matches everything
	  return true;
	if ( idx > 1 ) // remove repeated '*'
	  return isMatch(s, p+idx-1);

	for (int i = 0; i <= n; ++i)
	  if ( isMatch(s+i, p+1) ) return true;
      }

    return false;
  }

  // Dynamic programming solution, with O(n) space
  bool isMatch_v1(const char *s, const char *p)
  {
    if ( NULL == s || '\0' == s[0] )
      {
	if ( NULL == p || '\0' == p[0] ) return true;
	int idx = 0;
	while ( p[idx] != '\0' )
	  {
	    if ( p[idx++] != '*' )
	      return false;
	  }
	return true;
      }
    if ( NULL == p || '\0' == p[0] ) return false;

    int i = 0, k = 0; // for the initial run only
    for (; p[i] != '*' && p[i] != '\0' && s[k] != '\0'; ++i, ++k)
      {
	if ( p[i] != s[k] && p[i] != '?' )
	  return false;
      }
    if ( p[i] == '\0' ) return (s[k] == '\0');
    int cnt = 0;
    while ( p[i] == '*' ) { ++i; ++cnt; }
    if ( p[i] == '\0' ) return true;
    if ( cnt > 0 ) --i;
    s = s+k; p = p+i; // update the values
    
    // Start using DP only when the first char in p is '*'
    int n = strlen(s);
    bool *curr_tbl = new bool[n+1];
    bool *prev_tbl = new bool[n+1];
    bool res = false;

    prev_tbl[0] = true; // empty matches empty
    for (int k = 1; k <= n; ++k) prev_tbl[k] = false; // empty not match str
    curr_tbl[0] = (p[0] == '*'); // star matches empty
    bool skip_star = false; 
    for (int i = 0; p[i] != '\0'; ++i)
      {
	char ch = p[i];
	if ( '*' == ch )
	  {
	    if ( skip_star )
	      continue;
	    else
	      skip_star = true;

	    curr_tbl[0] = false;
	    int k;
	    for (k = 0; k <= n; ++k)
	      if ( prev_tbl[k] ) break;
	    for (; k <= n; ++k)
	      curr_tbl[k] = true;
	  }
	else if ( '?' == ch )
	  {
	    curr_tbl[0] = false; 
	    for (int k = 1; k <= n; ++k)
	      curr_tbl[k] = prev_tbl[k-1];
	  }
	else
	  {
	    curr_tbl[0] = false;
	    for (int k = 1; k <= n; ++k)
	      curr_tbl[k] = prev_tbl[k-1] && (ch == s[k-1]);
	  }

	skip_star = false;
	bool *tmp = curr_tbl;
	curr_tbl = prev_tbl;
	prev_tbl = tmp;
      }

    res = prev_tbl[n]; 
    delete [] curr_tbl;
    delete [] prev_tbl;
    return res;
  }

  bool isMatch(const char *s, const char *p)
  {
    const char *star = NULL;
    const char *ss = s;
    while ( *s )
      {
	// Normal match
	if ( (*p == '?') || (*p == *s) ) { ++s; ++p; continue; } 
	
	// So long as there is a star, matching continues
	// Only need to remember the last star's position
	if ( *p == '*' ) { star = p++; ss = s; continue; } 

	// Retract to the last star position
	if ( star ) { p = star+1; s = ++ss; continue; } 

	return false;
    }

    //check for remaining characters in pattern
    while ( *p == '*' ) ++p;
    return !*p;  
  }
};

int main()
{
  Solution sol;

#define test_case(S, P, val) {					\
    bool res = sol.isMatch((S), (P));				\
    printf("%s:\t %s\n", (S), (P));				\
    if ( res != (val) ) cout << "Error: mismatched" << endl; }	\

  test_case("aa", "a", false);
  test_case("aa", "aa", true);
  test_case("aaa", "aa", false);
  test_case("aa", "*", true);
  test_case("aa", "a*", true);
  test_case("ab", "?*", true);
  test_case("aab", "c*a*b", false);
  test_case("", "*", true);
  test_case("abefcdgiescdfimde", "ab*cd?i*de", true);  
}

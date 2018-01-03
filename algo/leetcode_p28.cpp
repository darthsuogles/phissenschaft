/**
 * LeetCode Problem 28
 *
 * Implement the function strStr()
 */

#include <iostream>

using namespace std;

class Solution {
public:
  // Using pure pointer arithmetic 
  int strStr(char *haystack, char *needle)
  {
    if ( haystack == NULL || needle == NULL ) return -1;
    char *tp = haystack;
    char *q = needle;
    for (; *tp != '\0' && *q != '\0';)
      {	
	if ( *tp == *q )
	  {
	    ++tp;
	    ++q;
	  }
	else
	  {
	    tp = tp - (q - needle) / sizeof(char) + 1;
	    q = needle;
	  }
      }
    
    if ( *q == '\0' )
      return (tp - (q - needle) - haystack) / sizeof(char);
    else
      return -1;
  }
};

int main()
{
  Solution sol;
#define test_case(T, Q) {			\
    char *text = (T);				\
    char *query = (Q);				\
    cout << sol.strStr(text, query) << endl; }	\


  test_case("mississippi", "issip");
}

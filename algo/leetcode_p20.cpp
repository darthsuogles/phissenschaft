#include <iostream>
#include <stack>
#include <cassert>
#include <string>

using namespace std;

class Solution {
public:
  bool isValid(string s)
  {
    stack<char> stk;
    for ( int i = 0; i < s.size(); ++i )
      {
	char ch = s[i];

	switch ( ch )
	  {
	  case '(':
	  case '[':
	  case '{':
	    stk.push(ch);
	    break;	    
	  case ')':
	    if ( stk.empty() || stk.top() != '(' )
	      return false;
	    stk.pop();
	    break;
	  case ']':
	    if ( stk.empty() || stk.top() != '[' )
	      return false;
	    stk.pop();
	    break;
	  case '}':
	    if ( stk.empty() || stk.top() != '{' )
	      return false;
	    stk.pop();
	    break;
	  }
      }

    return stk.empty();
  }
};

int main()
{
  Solution sol;

  assert( sol.isValid("{}") );
  assert( sol.isValid("{}[]()") );
  assert( sol.isValid("{()}[{}](()())") );
  assert( ! sol.isValid("{}[]()]") );
}

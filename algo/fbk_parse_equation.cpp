/**
 * FaceBook interview question
 *
 * Parse a linear equation of x
 */

#include <iostream>
#include <string>
#include <limits.h>

using namespace std;

float solve_equation(string eqn_str)
{
  int len = eqn_str.size();

  int idx = 0;
  // Skip the leading spaces
  for (; idx < len; ++idx)
    if ( idx != ' ' ) break;
  
  float a = 0.f, b = 0.f; // a * x = b
  bool is_x = false;
  float lhs = 1.f;
  float sign = 1.f;

  // Left hand side
  for (; idx < len; ++idx)
    {
      char ch = eqn_str[idx];
      if ( '-' == ch )
	sign = -1.f;
      else if ( '+' == ch )
	sign = 1.f;
      else if ( '=' == ch )
	lhs = -1.f;
      else if ( ' ' == ch )
	{
	  for (; idx < len; ++idx)
	    if ( eqn_str[idx] != ' ' )
	      {
		--idx; break;
	      }
	}
      else
	{
	  bool is_x = false;
	  float curr = 1.f;

	  for (; idx < len; ++idx)
	    {
	      ch = eqn_str[idx];
	      if ( 'x' != ch )
		{		
		  float val = 0.f;
		  for (; idx < len; ++idx)
		    {
		      ch = eqn_str[idx];
		      if ( ch < '0' || ch > '9') break;
		      val = val * 10 + int(ch - '0');	      
		    }
		  curr *= val;
		}
	      else
		{
		  is_x = true;
		  ++idx;
		}
	      
	      // Skip spaces preceeding the next symbol
	      for (; idx < len; ++idx)
		if ( eqn_str[idx] != ' ' ) break;
	      if ( len == idx ) break;
	      
	      if ( eqn_str[idx] != '*' ) break;

	      // Skip spaces preceeding the next number
	      ++idx;
	      if ( len == idx ) break;
	      if ( eqn_str[idx] == ' ' )
		for (; idx < len; ++idx)
		  if ( eqn_str[idx] != ' ' )
		    {
		      --idx; break;
		    }
	    }

	  curr *= lhs * sign;
	  // cout << curr;
	  // if ( is_x )
	  //   cout << ":x";
	  // cout << endl;
	  
	  if ( is_x )
	    a += curr;
	  else
	    b += curr;
	}
    }

  if ( a != 0.f )
    return -b / a;
  else if ( b < 0 )
    return INT_MAX;
  else
    return INT_MIN;
}

int main()
{
  cout << solve_equation("1 + 32 + 14 * 2 * x = 22 + 123 + 2 * x + 1") << endl;
}

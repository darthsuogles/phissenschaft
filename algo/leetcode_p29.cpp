/**
 * LeetCode Problem 29
 *
 * Divide two integers without multiplication, division or mod
 */

#include <iostream>
#include <limits.h>

using namespace std;

class Solution {
public:
  int divide(int dividend, int divisor)
  {
    if ( 0 == divisor ) return INT_MAX;
    if ( 1 == divisor ) return dividend; // don't cover the -1 case, might overflow
    if ( INT_MIN == divisor ) // the integer with largest abs
      {
	if ( INT_MIN == dividend )
	  return 1;
	else
	  return 0;
      }
    if ( 0 == dividend ) return 0;
    if ( dividend == divisor ) return 1;
    if ( divisor < 0 ) // thus consider only when divisor is positive
      {
	int res = divide(dividend, -divisor);
	if ( INT_MIN == res )
	  return INT_MAX;
	else
	  return -res;
      }
    
    int a = dividend;
    int b = divisor;
    int cnt = 0;    

    if ( a > 0 )
      {
	if ( a < b )
	  return 0;
	
	while ( true )
	  {
	    if ( a == b )
	      break;
	    if ( a < (b + b) )
		break;
	    if ( b > INT_MAX - b )
	      break;

	    b += b;
	    ++cnt;
	  }

	return (1 << cnt) + divide(a - b, divisor);
      }
    else // a < 0
      {
	b = -b;
	if ( a > b )
	  return 0;
	
	while ( true )
	  {
	    if ( a == b )
	      break;
	    if ( a > (b + b) )
	      break;
	    if ( b < INT_MIN - b )
	      break;

	    b += b;
	    ++cnt;
	  }
	return -(1 << cnt) + divide(a - b, divisor);
      }
  }
};

int main()
{
  Solution sol;

  cout << sol.divide(10, 2) << endl;
  cout << sol.divide(10, 3) << endl;
  cout << sol.divide(-10, 3) << endl;
  cout << sol.divide(INT_MIN, -1) << endl;
  cout << sol.divide(INT_MIN, 2) << endl;
}

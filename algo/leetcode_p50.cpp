/**
 * LeetCode Problem 50
 *
 * Power
 */

#include <iostream>
#include <cmath>

using namespace std;

class Solution {
public:
  // Recursively call
  double pow_v0(double x, int n)
  {
    if ( 0 == n ) return 1.0; // take 0^0 = 1
    if ( 1 == n ) return x;
    if ( 0.0 == x ) return 0.0;
    if ( n < 0 ) return pow(1/x, -n);
    int i = 1;
    double res = x;
    while ( (i << 1) <= n )
      {
	res *= res;
	i <<= 1;
      }
    return res * pow(x, n-i);
  }

  double pow(double x, int n)
  {
    if ( 0 == n ) return 1.0; // take 0^0 = 1
    if ( 1 == n ) return x;
    if ( 0.0 == x ) return 0.0;
    if ( 1.0 == x ) return 1.0;
    if ( n < 0 ) return pow(1/x, -(n+1))/x;

    double a = x;
    double res = 1.0;
    while ( n > 0 )
      {
	if ( n & 1 )
	  res *= a;
	a *= a;
	n >>= 1;
      }
    return res;
  }
};

int main()
{
  Solution sol;
#define test_case(a, n) cout << sol.pow(a, n) << " " << pow(a, n) << endl

  test_case(1.23, 30);
  test_case(34.00515, -3);
}

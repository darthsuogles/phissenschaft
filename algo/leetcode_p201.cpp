/**
 * LeetCode Problem 201
 *
 * Ranged bitwise AND
 */

#include <iostream>
#include <cmath>
#include <limits.h>

using namespace std;

class Solution {
public:
  int rangeBitwiseAnd(int m, int n)
  {
    if ( m == n ) return m;
    int res = m & n;
    int cnt = n - m + 1;    
    return res & (INT_MAX << int(ceil(log2(cnt))));
  }
};

int main()
{
  Solution sol;
#define test_case(m, n) cout << (m) << ", " << (n) << " : " << sol.rangeBitwiseAnd((m), (n)) << endl;
  
  test_case(0, 1);
  test_case(5, 7);
  test_case(2, 4);
  test_case(0, 2147483647);
  test_case(2147483646, 2147483647);
  test_case(20000, 2147483647);
}

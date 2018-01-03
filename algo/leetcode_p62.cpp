/**
 * LeetCode Problem 62
 *
 * Unique paths
 */

#include <iostream>

using namespace std;

class Solution {
public:
  int uniquePaths(int m, int n)
  {
    if ( m <= 1 || n <= 1 ) return 1;
    if ( m < n ) return uniquePaths(n, m);
    --m; --n;
    // Really just \binom(m+n, n)
    long res = 1;
    for (int k = m+n; k >= m+1; --k)
      res *= k;
    for (int k = 1; k <= n; ++k)
      res /= k;
    return (int)res;
  }
};

int main()
{
  Solution sol;
  cout << sol.uniquePaths(3,3) << endl;
  cout << sol.uniquePaths(10, 10) << endl;
}

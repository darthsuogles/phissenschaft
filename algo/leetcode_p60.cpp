/**
 * LeetCode Problem 60
 *
 * Permutation sequence
 */

#include <string>
#include <iostream>

using namespace std;

class Solution {
public:
  string getPermutation(int n, int k)
  {
    if ( 0 == n || 0 == k ) return "";
    if ( 1 == n ) return "1";
    string res;

    // Compute current position
    int factor = 1;
    for (int d = 1; d <= n-1; factor *= d++);

    int a = k / factor;
    int rem = k % factor;
    string nxt;
    if ( 0 == rem ) // the last for such digit
      rem = factor; 
    else
      ++a;
    nxt = getPermutation(n-1, rem);

    res += char(a + int('0'));
    for (int i = 0; i < n-1; ++i)
      {	
	int b = int(nxt[i] - '0');
	if ( b >= a )
	  res.push_back( nxt[i]+1 );
	else
	  res.push_back( nxt[i] );
      }

    return res;
  }
};

int main()
{
  Solution sol;
  for (int k = 1; k <= 6; ++k)
    cout << sol.getPermutation(3, k) << endl;
}

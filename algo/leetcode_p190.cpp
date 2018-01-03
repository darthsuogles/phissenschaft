/**
 * LeetCode Problem 190
 *
 * Reverse bits for a 32-bit integer
 */

#include <iostream>

using namespace std;

class Solution {
public:
  uint32_t reverseBits(uint32_t n)
  {
    uint32_t res = 0;
    short cnt = 0;
    while ( n > 0 )
      {
	res <<= 1;
	res ^= n & 1;
	n >>= 1;
	++cnt;
      }
    if ( res > 0 )
      res <<= (32 - cnt);
    return res;
  }
};

int main()
{
  Solution sol;
  uint32_t res = sol.reverseBits(43261596);
  cout << (res == 964176192) << endl;
}

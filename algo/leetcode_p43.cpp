/**
 * LeetCode Problem 42
 *
 * Multiply two strings
 */

#include <iostream>
#include <string>

using namespace std;

class Solution {
public:
  string multiply(string num1, string num2)
  {
    if ( num1 == "0" || num2 == "0" ) return "0";
    string res((num1.size() + num2.size() + 1), '0'); // initialize to all zero

    for (int j = num2.size() - 1, shft = 0; j >= 0; --j, ++shft)
      {
	int d = int(num2[j] - '0');
	if ( 0 == d ) continue;
	int carry = 0;

	int k = shft;
	for (int i = num1.size() - 1; i >= 0; --i, ++k)
	  {
	    int a = int(num1[i] - '0');
	    int b = int(res[k] - '0');
	    int curr = a * d + b + carry;
	    carry = curr / 10;
	    res[k] = char((curr % 10) + int('0'));
	  }
	while ( carry > 0 )
	  {
	    int b = int(res[k] - '0');
	    int curr = b + carry;
	    carry = curr / 10;
	    res[k++] = char((curr % 10) + int('0'));
	  }
      }

    int bnd = res.find_last_not_of('0');
    if ( bnd < res.size() ) bnd += 1;
    res.resize(bnd);
    for (int i = 0, j = bnd-1; i < bnd/2; ++i, --j)
      {
	int tmp = res[i]; res[i] = res[j]; res[j] = tmp;
      }
    return res;
  }
};

int main()
{
  Solution sol;
  cout << sol.multiply("12", "12") << endl;
  cout << sol.multiply("5", "42") << endl;
}

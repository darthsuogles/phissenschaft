/**
 * LeetCode Problem 6: zig-zag string traversal
 */
#include <string>
#include <iostream>
#include <cassert>

using namespace std;

class Solution {
public:
  string convert(string s, int nRows) {
    if (nRows <= 1 || s.size() <= nRows)
      return s;

    int len = s.size();
    string conv_str(s.size(), '\0');

    int idx = 0;
    int stride = nRows + nRows - 2;
    int row = 0;
    for (int i = row; i < len; i += stride)
      conv_str[idx++] = s[i];
    ++row;

    int row_stride = stride - 2;
    for (; row < nRows - 1; ++row)
      {
	for (int i = row; i < len; i += stride)
	  {
	    conv_str[idx++] = s[i];
	    if ( i + row_stride < len )
	      conv_str[idx++] = s[i + row_stride];
	  }
	row_stride -= 2;
      }

    for (int i = row; i < len; i += stride)
      conv_str[idx++] = s[i];
    
    return conv_str;
  }
};


int main()
{
  Solution sol;
#define test_case(s, nRows)  cout << sol.convert((s), (nRows)) << endl

  test_case("PAYPALISHIRING", 3);
  test_case("PAYPALISHIRING", 3);
  test_case("AB", 1);
  test_case("ABC", 2);
  test_case("ABC", 20);
  test_case("ABCDABCDABCDD", 3);
  test_case("ABDECABDECABDEC", 4);
  test_case("ABCD", 2);
  test_case("ABCDE", 4);
  
  return 0;
}

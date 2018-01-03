#include <iostream>
#include <cassert>
#include <limits.h>

using namespace std;

int reverse(int x) {
    int sign = (x > 0) ? 1 : (-1);
    int a = x * sign;
    int b = 0;
    int INT_MAX = 1<<31;
    while (a > 0)
    {
      if ( b > INT_MAX / 10 ) return 0;
      b *= 10;
      int dec = a % 10;
      if ( b > INT_MAX - dec ) return 0;
      b += dec;
      a /= 10;
    }
    
    return sign * b;
}

int main()
{
#define test_case(a, b) {   \
    int res = reverse((a));				\
    cout << reverse((a)) << endl;			\
    assert(res == (b)); } 

  test_case(123, 321);
  test_case(-123, -321);
  test_case(1, 1);
  test_case(1000000003, 0);
  test_case(10, 1);
  test_case(100, 1);
  test_case(1534236469, 0);

  return 0;
}

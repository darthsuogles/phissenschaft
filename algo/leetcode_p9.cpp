/**
 * LeetCode Problem 9
 *
 *  Determine whether an integer is a palindrome. Do this without extra space.
 */

#include <limits.h>
#include <cassert>

bool isPalindrome(int x)
{
  if ( x < 0 ) return false;
  int a = x, b = 0;

  while ( a > 0 )
    {
      if ( b > INT_MAX / 10 ) return false;
      b *= 10;

      int dec = a % 10;
      if ( b > INT_MAX - dec ) return false;
      b += dec;
      a /= 10;
    }

  return ( b == x );
}

void test_case(int a, bool is_palindrome)
{
  assert( isPalindrome(a) == is_palindrome );
}

int main()
{
  test_case(121, true);
  test_case(122, false);
  test_case(12345, false);

  return 0;
}

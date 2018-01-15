#include <iostream>
#include <cassert>
#include <cstring>
#include <limits.h>

using namespace std;

int myAtoi(char *str)
{
  if ( NULL == str ) return 0;
  int len = strlen(str);
  char *cptr = str;
  for (; (*cptr == ' ' || *cptr == '\t') && (*cptr != '\0'); ++cptr);  
  
  int res = 0;
  
  if ( *cptr == '-' )
    {
      ++cptr;
      
      for (; *cptr != '\0'; ++cptr)
	{
	  int dec = (int)(*cptr - '0');
	  if ( dec < 0 || dec > 9 ) break;

	  if ( res < INT_MIN / 10 ) return INT_MIN;
	  res *= 10;	  
	  if ( res < INT_MIN + dec ) return INT_MIN;
	  res -= dec;
	}

      return res;
    }
  else if ( *cptr == '+' )
    ++cptr;
  
  for (; *cptr != '\0'; ++cptr)
    {
      int dec = (int)(*cptr - '0');
      if ( dec < 0 || dec > 9 ) break;

      if ( res > INT_MAX / 10 ) return INT_MAX;
      res *= 10;
      if ( res > INT_MAX - dec ) return INT_MAX;
      res += dec;
    }

  return res;
}

int main()
{
#define test_case(str, a) {			\
    int res = myAtoi((str));			\
    cout << res << endl;			\
    assert(res == (a)); }

  test_case("123", 123);
  test_case("  123", 123);
  test_case("  +123", 123);
  test_case("  -123", -123);
  test_case(" 23000000001", INT_MAX);
  test_case("  9534236469", INT_MAX);
  test_case(" -23000000001", INT_MIN);
  test_case("", 0);
  test_case("  ", 0);
  test_case(" 123asd1231", 123);
  test_case(" +", 0);
  test_case(" +sds", 0);
  
  return 0;
}

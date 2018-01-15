#include <iostream>
#include <string>

using namespace std;

inline bool isVowel(char ch) {
  switch (ch) {
  case 'a':
  case 'e':
  case 'i':
  case 'o':
  case 'u':
    return true;
  default:
    return false;
  }
}

void revStrVowels(string &s) {
  int n = s.size();
  if ( 0 == n ) return; 
  int i = 0, j = n - 1;
  while ( i < j ) {
    for (; i < n; ++i)
      if ( isVowel(s[i]) ) break;
    if ( n == i ) break;
    for (; j > i; --j)
      if ( isVowel(s[j]) ) break;
    if ( j == i ) break;
    char ch = s[i];
    s[i] = s[j];
    s[j] = ch;
    ++i; --j;
  } 
}

void testCase(string s) {
  cout << s << endl;
  revStrVowels(s);
  cout << s << endl;
}

int main() {

  testCase("united states");
  testCase("");
  testCase("bcd");
  testCase("ade");
}

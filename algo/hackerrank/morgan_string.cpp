/**
 * 
 * https://www.hackerrank.com/challenges/morgan-and-a-string
 */

#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;


int main() {
  /* Enter your code here. Read input from STDIN. Print output to STDOUT */
  int N;
  cin >> N;

  while ( N-- > 0 ) {
    string A, B;
    cin >> A;
    cin >> B;
    size_t len_A = A.size(), len_B = B.size();
    char chA_end = char('Z' + 1);
    char chB_end = char('Z' + 2);
    A.push_back(chA_end);
    B.push_back(chB_end);
    
    int i = 0, j = 0;
    for (; i < len_A && j < len_B; ) {
      char chA = A[i];
      char chB = B[j];
      if ( chA == chB ) { // must look ahead
	int cmp = A.compare(i+1, len_A - i, B, j+1, len_B - j);
	if ( cmp < 0 ) {
	  cout << chA; ++i; 
	} else {
	  cout << chB; ++j;
	}
      }
      else if ( chA < chB ) {
	cout << chA;
	++i;
      } else {
	cout << chB;
	++j;
      }
    }
    if ( i < len_A ) {
      for (; i < len_A; ++i) {
	cout << A[i];
      }
    }
    else {
      for (; j < len_B; ++j) {
	cout << B[j];
      }
    }
    cout << endl;
  }
  
  return 0;
}


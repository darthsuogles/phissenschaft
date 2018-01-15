#include <iostream>
#include <vector>

using namespace std;

void findMult23(int N) {
  int p = 1, q = 1;
  while (true) {
    int a = 2*p, b = 3*q;
    if ( a < b ) {
      if ( a > N ) break;
      cout << a << ": " << 2;
      if ( 0 == p % 3 )
	cout << ", " << 3 << endl;
      else
	cout << endl;
      ++p;
    } else {
      if ( b > N ) break;
      cout << b << ": " << 3 << endl;
      ++q; if ( 0 == (q & 1) ) ++q;
    }
  }
  cout << "-----------------" << endl;
}

void findFact23(int N) {
  vector<int> res;
  res.push_back(1);
  vector<int> factors = {2,3,5,7,13};
  int K = factors.size();
  vector<int> inds(K, 0);  
  while (true) {
    int val = 1 + N;
    for (int i = 0; i < K; ++i) {
      val = min(val, factors[i] * res[inds[i]]);
    }
    if ( val > N ) break;
    cout << val << ": ";
    for (int i = 0; i < K; ++i) {
      if ( val == factors[i] * res[inds[i]] ) {
	cout << factors[i] << " "; 
	++inds[i];
      }
    }
    cout << endl;
    res.push_back(val);
  }
}

int main() {
  findMult23(7);
  findMult23(20);
  findFact23(40);
}

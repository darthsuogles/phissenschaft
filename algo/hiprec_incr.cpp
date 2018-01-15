#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

vector<int> increment(vector<int> num) {
  vector<int> res;
  if ( num.empty() ) return res;
  int carry = 1;
  int i = num.size() - 1;
  for (; i >= 0; --i) {
    int val = num[i] + carry;
    switch ( val ) {
    case 10:
      res.push_back(0);
      break;
    default:
      carry = 0;
      res.push_back(val);
      break;
    }
    if ( 0 == carry ) {
      for (int j = i-1; j >= 0; --j)
	res.push_back(num[j]);
      break;
    }
  }
  if ( 1 == carry )
    res.push_back(1);
  reverse(res.begin(), res.end());
  return res;
}

void testCase(vector<int> num) {
  vector<int> res = increment(num);
  for (auto it = res.begin(); it != res.end(); ++it)
    cout << *it << " ";
  cout << endl;
  cout << "-----------" << endl;
}

int main() {
  testCase({1,2,3,4});
  testCase({1,2,3,9});
  testCase({1});
  testCase({});
  testCase({9});
  testCase({9,9,9});
}

#include <iostream>
#include <vector>

using namespace std;

int main() {
  int T; cin >> T;
  for (int idx = 1; idx <= T; ++idx) {
    int N; cin >> N; // number of integers 
    int P; cin >> P; // price of the target
    vector<int> arr(N);
    for (int i = 0; i < N; ++i)
      cin >> arr[i];

    long cntNumSeq = 0;
    long partialSum = 0; // ! this was mistakenly taken as int
    int i = 0, j = 0;
    for (; j < N; ++j) {
      int val = arr[j];
      if ( val >= P ) {
	if ( val == P )
	  ++cntNumSeq;
	partialSum = 0;
	i = j+1; continue;	
      }
      partialSum += arr[j];
      if ( partialSum > P ) {
	for (; i < j; ++i) {
	  if ( partialSum <= P ) break;
	  partialSum -= arr[i];
	}
      }
      cntNumSeq += j - i + 1;      
    }

    cout << "Case #" << idx << ": " << cntNumSeq << endl;
  }
}

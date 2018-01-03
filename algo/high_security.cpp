#include <iostream>
#include <vector>

using namespace std;

int main() {
  int T; cin >> T;
  for (int idx = 1; idx <= T; ++idx) {
    int N; cin >> N;
    const int M = 2;
    vector< vector<char> > rows(M, vector<char>(N+2));
    for (int d = 0; d < M; ++d) {
      rows[d][0] = 'X';
      for (int i = 1; i <= N; ++i) cin >> rows[d][i];
      rows[d][N+1] = 'X';
    }

    int nGuards = 0;
    for (int d = 0; d < M; ++d) {
      auto &curr = rows[d];
      auto &next = rows[(d+1) % M];
      for (int i = 1; i <= N; ++i) {
	if ( 'X' == curr[i] ) continue;
	if ( 'X' == curr[i-1] && 'X' == curr[i+1] ) {
	  if ( 'X' == next[i] ) continue;
	  next[i] = '@';
	  curr[i] = '#';
	}
      }
    }

    for (int d = 0; d < M; ++d) {
      auto &curr = rows[d];
      bool isBlocked = true;
      int cntSingleton = 0;
      for (int i = 0; i <= N+1; ++i) {
	char ch = curr[i];
	if ( 'X' == ch || '#' == ch ) {
	  if ( ! isBlocked ) {
	    isBlocked = true;
	    if ( 0 == cntSingleton )
	      ++nGuards;
	    else
	      nGuards += cntSingleton;
	  }
	  continue;
	}
	if ( isBlocked ) {
	  cntSingleton = 0;
	  isBlocked = false;
	}
	if ( '@' == ch ) 
	  ++cntSingleton;
      }
    }
    
    cout << "Case #" << idx << ": " << nGuards << endl;
  }
}

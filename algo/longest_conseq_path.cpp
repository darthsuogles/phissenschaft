#include <iostream>
#include <vector>

using namespace std;

int maxlen_conseq_path(const int i, const int j,
		       const vector< vector<int> > &A,
		       vector< vector<bool> > &is_visited) {
  int n = A.size();
  int max_len = 1;
  
  for (int ii = max(i-1, 0); ii <= min(n-1, i+1); ++ii)
    for (int jj = max(j-1, 0); jj <= min(n-1, j+1); ++jj) {
      if ( i == ii && j == jj ) continue;
      if ( i != ii && j != jj ) continue;
      if ( is_visited[ii][jj] ) continue;
      if ( A[ii][jj] != A[i][j] + 1 ) continue;
      is_visited[ii][jj] = true;
      max_len = max(max_len, 1 + maxlen_conseq_path(ii, jj, A, is_visited));
      is_visited[ii][jj] = false;
    }
  return max_len;
}

int longest_conseq_path(const vector< vector<int> > &A) {
  if ( A.empty() ) return 0;
  int n = A.size();
  if ( A[0].empty() ) return 0;

  int max_len = 0;
  vector< vector<bool> > is_visited(n, vector<bool>(n, false));
  
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {      
      is_visited[i][j] = true;
      max_len = max(max_len, maxlen_conseq_path(i, j, A, is_visited));
      is_visited[i][j] = false;
    }
  return max_len;
}

int main() {
  vector< vector<int> > A = {
    {1,3,5},
    {2,4,6},
    {9,8,7}
  };

  cout << longest_conseq_path(A) << endl;
}

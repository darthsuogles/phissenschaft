#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

int main()
{
  /* Enter your code here. Read input from STDIN. Print output to STDOUT */
  while ( ! cin.eof() )
    {
      int N, M;
      cin >> N >> M;
      if ( N == 0 || M == 0 ) { cout << 0 << endl; return 0; }
  
      vector<int> coins(M);
      for (int i = 0; i < M; ++i) cin >> coins[i];
    
      vector< vector<long> > tbl(M, vector<long>(N+1, 0));

      int val = coins[0];
      for (int d = val; d <= N; d += val) tbl[0][d] = 1;
      for (int i = 1; i < M; ++i)
	{        
	  int val = coins[i];
	  for (int d = 1; d <= N; ++d)
	    {
	      long comb = tbl[i-1][d];
	      for (int k = val; k <= d; k += val)
		comb += tbl[i-1][d-k];
	      if ( 0 == (d % val) ) ++comb; // use only current coin
	      tbl[i][d] = comb;
	    }
	}
      cout << tbl[M-1][N] << endl;
    }

  return 0;
}

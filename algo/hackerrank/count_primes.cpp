#include <iostream>
#include <vector>

using namespace std;

/*
 * Complete the function below.
 */
int getNumberOfPrimes(int n) {
    vector<bool> sieve(n, false);
    for (int d = 2; d < n; ++d)
    {
        if ( sieve[d] ) continue;
        for (int k = 2*d; k < n; k += d) sieve[k] = true;
    }

    int cnt = 0;
    for (int d = 2; d < n; ++d) 
        if ( ! sieve[d] ) ++cnt;
    return cnt;
}

int main()
{
#define test_case(N) cout << getNumberOfPrimes((N)) << endl;

  test_case(2);
  test_case(3);
  test_case(10);
}

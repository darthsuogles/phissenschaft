// Author: Miguel Oliveira
#include <iostream>
#include <cstdio>

using namespace std;

const int MAX = 101000;
int price[MAX];

long long Solve() {
  int n, p;
  scanf("%d %d", &n, &p);
  for (int i = 0; i < n; ++i)
    scanf("%d", &price[i]);
  price[n] = p+1;

  long long ans = 0, sum = 0;
  int start = -1;
  for (int i = 0; i <= n; ++i) {
    sum += price[i];
    while (sum > p) {
      sum -= price[++start];
    }
    ans += i - start;
  }
  return ans;
}

int main() {
  int ntests;
  scanf("%d", &ntests);
  for (int nt = 1; nt <= ntests; ++nt) {
    printf("Case #%d: %lld\n", nt, Solve());
  }
  return 0;
}

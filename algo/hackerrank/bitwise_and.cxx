#include <iostream>

using namespace std;

int main() {
    int T; cin >> T;
    while (--T >= 0) {
        int N, K; cin >> N >> K;
        // First, let `T = (K - 1) | K`
        // `K <= T` and the equality holds if K is odd
        // `(T & (K - 1)) == (K - 1)`
        // Since the problem assumes K <= N, if K is even,
        // then `K - 1` is odd and `(K - 2) | (K - 1)` is `K - 1`.
        // No matter what, the check `T <= N` will tell us
        // if `K - 1` could be the best choice.
        int res = ((K - 1) | K) <= N ? (K - 1) : (K - 2);
        cout << res << endl;
    }
}

#include <iostream>
#include <vector>
#include <stack>
#include <cmath>

using namespace std;

class Prime {
    vector<int> prime_;

    void fill_primes_(int n) {
        if (n < 2) return;
        int bnd = int(2 * n * log(n) + 7);
        vector<bool> sieve(bnd + 1, false);
        for (int a = 2; a <= bnd; ++a) {
            if (sieve[a]) continue;
            for (int b = a + a; b <= bnd; b += a) {
                sieve[b] = true;
            }
        }
        for (int i = 2; i <= bnd; ++i) {
            if (!sieve[i]) prime_.push_back(i);
        }
    }

public:
    Prime(int n) { fill_primes_(n); }
    inline int get(int k) { return prime_[k]; }
    inline size_t size() { return prime_.size(); }
};

auto prime = Prime(40000);

int main() {
    int N, Q; cin >> N >> Q;
    vector<stack<int>> A(Q + 1);
    vector<stack<int>> B(Q + 1);
    for (int i = 0; i < N; ++i) {
        int a; cin >> a;
        A[0].push(a);
    }
    for (int i = 1; i <= Q; ++i) {
        auto &pile = A[i-1];
        while (!pile.empty()) {
            int a = pile.top(); pile.pop();
            int p = prime.get(i - 1);
            if (0 == a % p) B[i].push(a); else A[i].push(a);
        }
    }
    for (int i = 1; i <= Q; ++i) {
        auto &pile = B[i];
        while (!pile.empty()) {
            cout << pile.top() << endl;
            pile.pop();
        }
    }
    auto &pile = A[Q];
    while (!pile.empty()) {
        cout << pile.top() << endl;
        pile.pop();
    }
}

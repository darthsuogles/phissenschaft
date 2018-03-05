#include <iostream>
#include <vector>
#include <queue>
#include <cstdio>
#include <cassert>

using namespace std;

using real = double;
using min_heap_t = priority_queue<real, vector<real>, greater<real>>;
using max_heap_t = priority_queue<real, vector<real>, less<real>>;

int main() {
    int n; cin >> n;
    // Elements first come here, want the smallest on top
    min_heap_t in_heap;
    // Pop the smaller half there, want the largest on top
    max_heap_t out_heap;

    real a; cin >> a;
    if (0 == in_heap.size()) {
        in_heap.push(a);
        printf("%.01f\n", in_heap.top());
    }

    for (int i = 1; i < n; ++i) {
        cin >> a;
        a < in_heap.top() ? out_heap.push(a) : in_heap.push(a);

        if (in_heap.size() > out_heap.size() + 1) {
            out_heap.push(in_heap.top());
            in_heap.pop();
        } else if (in_heap.size() + 1 < out_heap.size()) {
            in_heap.push(out_heap.top());
            out_heap.pop();
        }

        if (in_heap.size() == out_heap.size()) {
            printf("%.01f\n", (in_heap.top() + out_heap.top()) / 2.0);
        } else if (in_heap.size() > out_heap.size()) {
            printf("%.01f\n", in_heap.top());
        } else {
            printf("%.01f\n", out_heap.top());
        }
    }
}

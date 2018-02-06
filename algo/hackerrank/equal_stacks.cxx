#include <iostream>
#include <vector>
#include <queue>
#include <utility>
#include <climits>
#include <tuple>

using namespace std;

int main() {
    const int M = 3;
    int n[M];
    for (int i = 0; i < M; cin >> n[i++]);
    vector<vector<int>> arrs;

    int sum[M] = {0};
    int a;
    for (int idx = 0; idx < M; ++idx) {
        vector<int> arr(n[idx]);
        for (int i = arr.size() - 1; i >= 0; --i) {
            cin >> a;
            sum[idx] += (arr[i] = a);
        }
        arrs.push_back(arr);
    }

    // Initialize the max-heap
    int min_h = INT_MAX;
    priority_queue<tuple<int, int, int>> hi_stack;
    for (int idx = 0; idx < M; ++idx) {
        min_h = min(sum[idx], min_h);
        hi_stack.push(make_tuple(sum[idx], idx, arrs[idx].size() - 1));
    }

    while (!hi_stack.empty()) {
        if (0 == min_h) break;
        auto curr_max_hidx = hi_stack.top();
        hi_stack.pop();
        int h = get<0>(curr_max_hidx);
        int idx = get<1>(curr_max_hidx);
        int j = get<2>(curr_max_hidx);
        if (h == min_h) break;
        h -= arrs[idx][j];
        min_h = min(h, min_h);
        hi_stack.push(make_tuple(h, idx, j - 1));
    }
    cout << min_h << endl;
}

#include <iostream>
#include <vector>
#include <queue>
#include <utility>

using namespace std;

class Solution {
    using int2 = pair<int, int>;
    using min_heap_t = priority_queue<int2, vector<int2>, greater<int2>>;
public:
    int kthSmallest(vector<vector<int>>& matrix, int k) {
        int m = matrix.size();
        if (0 == m) return -1;
        int n = matrix.size();
        if (0 == n) return -1;
        min_heap_t next_vals;
        for (int i = 0; i < m; ++i) {
            next_vals.push(make_pair(matrix[i][0], i));
        }
        vector<int> curr_col(m, 0);
        while (!next_vals.empty()) {
            int val = get<0>(next_vals.top());
            int row = get<1>(next_vals.top());
            next_vals.pop();
            if (--k == 0) return val;
            int col = ++curr_col[row];
            if (col == n) continue;
            next_vals.push(make_pair(matrix[row][col], row));
        }
        return -1;
    }
};

Solution sol;

void TEST(vector<vector<int>> matrix, int k) {
    cout << sol.kthSmallest(matrix, k) << endl;
}

int main() {
    TEST({{1, 5, 9}, {3, 7, 13}, {10, 13, 15}}, 8);
}

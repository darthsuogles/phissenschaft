#include <iostream>
#include <vector>

using namespace std;

class MatrixOptimizer {
    vector<vector<int>> matrix;

public:

    MatrixOptimizer(int n): matrix(n, vector<int>(n)) {
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                cin >> matrix[i][j];

        int mid = n / 2;

        long upper_sum = 0;
        for (int i = 0; i < mid; ++i) {
            for (int j = 0; j < mid; ++j) {
                upper_sum +=
                    max(max(matrix[n - i - 1][j], matrix[i][n - j - 1]),
                        max(matrix[i][j], matrix[n - i - 1][n - j - 1]));
            }
        }
        cout << upper_sum << endl;
    }
};

int main() {
    int q; cin >> q;
    for (; q > 0; --q) {
        int n; cin >> n;
        MatrixOptimizer(n + n);
    }
}

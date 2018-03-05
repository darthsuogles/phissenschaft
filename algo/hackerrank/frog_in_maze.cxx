#include <iostream>
#include <utility>
#include <cmath>
#include <vector>

using namespace std;

using real = long double;

// Standard Gaussian elimination
bool gauss_solve(vector<vector<real>> &A, vector<real> &x, const real eps = 1e-12) {
	int n = A.size(), m = A[0].size() - 1;

    const int SKIP_ROW = -1;
	vector<int> where(m, SKIP_ROW);
	for(int col = 0, row = 0; col < m && row < n; col++) {
        int sel = row;
        for (int i = row; i < n; i++) {
            if (abs(A[i][col]) > abs(A[sel][col])) sel = i;
        }
        if (abs(A[sel][col]) < eps) continue;

        for (int i = col; i <= m; ++i)
            swap(A[sel][i], A[row][i]);
        where[col] = row;

        for (int i = 0; i < n; i++) {
            if (i == row) continue;
            if (abs(A[i][col]) < eps) continue;
            auto c = A[i][col] / A[row][col];
            for(int j = 0; j <= m; j++)
                A[i][j] -= c * A[row][j];
        }
        row++;
    }

    for(int i = 0; i < m; i++) {
        if (SKIP_ROW == where[i]) continue;
        x[i] = A[where[i]][m] / A[where[i]][i];
    }

    // Check result
    for(int i = 0; i < n; ++i) {
        real sum = A[i][m];
        for(int j = 0; j < m; j++)
            sum -= x[j] * A[i][j];

        if (abs(sum) > eps) return false;
    }
    return true;
}

int main() {
    int n, m, num_channels; cin >> n >> m >> num_channels;
    vector<vector<char>> maze(n, vector<char>(m));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            cin >> maze[i][j];
        }
    }

    // Construct the channels
    const int NO_CHANNEL = -1;
    vector<vector<int>> channel_inds(n, vector<int>(m, NO_CHANNEL));
    vector<pair<int, int>> channel_ends(2 * num_channels);
    for (int k = 0; k < num_channels; ++k) {
        int i1, j1, i2, j2;
        cin >> i1 >> j1 >> i2 >> j2;
        --i1; --j1; --i2; --j2;

        int idx = 2 * k;
        channel_inds[i1][j1] = idx;
        channel_ends[idx] = make_pair(i2, j2);
        ++idx;
        channel_inds[i2][j2] = idx;
        channel_ends[idx] = make_pair(i1, j1);
    }

#define IDX(i, j) ((i) * m + (j))

    int init_i, init_j;

    // Construct a linear system Ax = 0
    int N = n * m;
    vector<vector<real>> A(N, vector<real>(N + 1, 0.0));
    vector<real> prob(N + 1, 0.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            char ch = maze[i][j];
            int idx = IDX(i, j);
            if ('#' == ch) continue;
            A[idx][idx] = 1.0;
            if ('*' == ch) continue;
            if ('%' == ch) {
                A[idx][N] = 1.0;
                continue;
            }
            if ('A' == ch) {
                init_i = i; init_j = j;
            }

            int ie = i, je = j;
            int cidx = channel_inds[i][j];
            if (-1 != cidx) {
                ie = get<0>(channel_ends[cidx]);
                je = get<1>(channel_ends[cidx]);
            }
            vector<int> adj_list;
            for (int ix = max(0, ie - 1); ix <= min(n - 1, ie + 1); ++ix) {
                for (int jx = max(0, je - 1); jx <= min(m - 1, je + 1); ++jx) {
                    if ((ix != ie) == (jx != je)) continue;
                    if ('#' == maze[ix][jx]) continue;
                    adj_list.push_back(IDX(ix, jx));
                }
            }
            for (auto pos: adj_list) {
                A[idx][pos] = -1.0 / adj_list.size();
            }
        }
    }

    gauss_solve(A, prob);
    cout << prob[IDX(init_i, init_j)] << endl;
}

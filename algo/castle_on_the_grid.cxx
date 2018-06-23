#include <iostream>
#include <vector>
#include <queue>

using namespace std;

class CastleMover {
    const int n;
    vector<vector<char>> grid;
    vector<vector<int>> steps;
    queue<int> candidates;

    bool fill_next(int i, int j, int curr_steps) {
        if ('X' == grid[i][j]) return false;
        // Don't stop if we hit a visited cell
        if (-1 == steps[i][j]) {
            steps[i][j] = curr_steps + 1;
            candidates.push(i * n + j);
        }
        return true;
    }

public:
    CastleMover(int n)
        : n(n), grid(n, vector<char>(n)), steps(n, vector<int>(n, -1)) {}

    int run() {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                cin >> grid[i][j];
            }
        }
        int x0, y0, x1, y1;
        cin >> x0 >> y0 >> x1 >> y1;

        steps[x0][y0] = 0;
        candidates.push(x0 * n + y0);
        while (!candidates.empty()) {
            int idx = candidates.front();
            candidates.pop();
            int x = idx / n; int y = idx % n;
            int curr_steps = steps[x][y];
            if (x1 == x && y1 == y) return curr_steps;
            // Search directions

            for (int i = x - 1;
                 i >= 0 && fill_next(i, y, curr_steps); --i);
            for (int i = x + 1;
                 i < n && fill_next(i, y, curr_steps); ++i);
            for (int j = y - 1;
                 j >= 0 && fill_next(x, j, curr_steps); --j);
            for (int j = y + 1;
                 j < n && fill_next(x, j, curr_steps); ++j);
        }
        return -1;
    }
};

int main() {
    int n; cin >> n;
    cout << CastleMover(n).run() << endl;
}

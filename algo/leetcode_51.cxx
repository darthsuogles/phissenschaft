#include <iostream>
#include <vector>
#include <string>

using namespace std;

class Solution {
    vector<vector<string>> solution_boards;

    void solve(const int col, vector<vector<bool>> &has_queen) {
        const int N = has_queen.size();
        if (col == N) {
            // Add this solution and exit
            vector<string> board;
            for (auto row: has_queen) {
                string curr;
                for (int i = 0; i < N; ++i) {
                    curr += row[i] ? "Q" : ".";
                }
                board.push_back(curr);
            }
            solution_boards.push_back(board);
            return;
        }
        // Check each cell in this column
        for (int i = 0; i < N; ++i) {
            // Check horizontal
            bool has_horizontal = false;
            for (int j = 0; j < col; ++j) {
                if (!has_queen[i][j]) continue;
                has_horizontal = true; break;
            }
            if (has_horizontal) continue;

            // Check diagonal
            bool has_diagonal = false;
            for (int x = i, y = col; x >= 0 && y >= 0; --x, --y) {
                if (!has_queen[x][y]) continue;
                has_diagonal = true; break;
            }
            if (has_diagonal) continue;

            // Check anti-diagonal
            for (int x = i, y = col; x < N && y >= 0; ++x, --y) {
                if (!has_queen[x][y]) continue;
                has_diagonal = true; break;
            }
            if (has_diagonal) continue;

            // Check if further solutions are admissible
            has_queen[i][col] = true;
            solve(col + 1, has_queen);
            has_queen[i][col] = false;
        }
    }

public:
    vector<vector<string>> solveNQueens(int N) {
        solution_boards.clear(); // make this thread-local
        vector<vector<bool>> has_queen(N, vector<bool>(N, false));
        solve(0, has_queen);
        return solution_boards;
    }
};

Solution sol;

void TEST(int N) {
    cout << endl << "TESTING: " << N << endl;
    auto res = sol.solveNQueens(N);
    for (auto board: res) {
        for (auto row: board) cout << row << endl;
        cout << "-----------------" << endl;
    }
}

int main() {
    TEST(4);
    TEST(8);
}

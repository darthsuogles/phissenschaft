/**
 * Number of Islands
 *
 * Given a 2d grid map of '1's (land) and '0's (water), count the number of islands.
 * An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.
 * You may assume all four edges of the grid are all surrounded by water.
 */
#include <iostream>
#include <cassert>
#include <unordered_set>
#include <vector>

using namespace std;

class UnionFind {
    vector<int> parent;
    vector<int> rank;
    int node_counts;  // the graph is sparse
    const int N;
public:
    UnionFind(int N): rank(N, 0), parent(N), N(N) {
        for (int i = 0; i < N; parent[i] = i, ++i);
    }

    int find_parent(int u) {
        assert(u < N);
        int pu = parent[u];
        if (pu != parent[pu])
            pu = find_parent(pu);
        return parent[u] = pu;
    }

    // Return 1 if merging causes the number of components to decrease by one
    int merge(int u, int v) {
        assert(u < N && v < N);
        int pu = find_parent(u);
        int pv = find_parent(v);
        if (pu == pv) return 0;
        int pu_rank = rank[pu];
        int pv_rank = rank[pv];
        if (pu_rank < pv_rank) {
            parent[pu] = pv;
        } else if (pu_rank > pv_rank) {
            parent[pv] = pu;
        } else {
            parent[pu] = pv;
            ++rank[pv];
        }
        return 1;
    }
};

class Solution {
    // Write over the original board
    void paint_over(vector<vector<char>> &grid, int i, int j) {
        switch (grid[i][j]) { case '$': case '0': return; }
        grid[i][j] = '$';
        int m = grid.size(), n = grid[0].size();
        if (i + 1 < m)
            paint_over(grid, i + 1, j);
        if (i >= 1)
            paint_over(grid, i - 1, j);
        if (j + 1 < n)
            paint_over(grid, i, j + 1);
        if (j >= 1)
            paint_over(grid, i, j - 1);
    }

public:
    int numIslands(vector<vector<char>> &grid) {
        int m = grid.size();
        if (0 == m) return 0;
        int n = grid[0].size();
        if (0 == n) return 0;

        UnionFind dset(m * n);
        int node_cnts = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if ('1' != grid[i][j])
                    continue;
                ++node_cnts;
                int u = i * n + j;
                // Only need to check the left and upper cells
                if (j > 0 && '1' == grid[i][j - 1])
                    node_cnts -= dset.merge(u, u - 1);
                if (i > 0 && '1' == grid[i - 1][j])
                    node_cnts -= dset.merge(u, u - n);
            }
        }
        return node_cnts;
    }

    int numIslandsPaintOver(vector<vector<char>> &grid) {
        int m = grid.size();
        if (0 == m) return 0;
        int n = grid[0].size();
        if (0 == n) return 0;
        int num_islands = 0;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                if ('1' == grid[i][j]) {
                    ++num_islands;
                    paint_over(grid, i, j);
                }
            }
        return num_islands;
    }
};

Solution sol;

void TEST(vector<string> rows, int expected) {
    vector<vector<char>> grid;
    for (auto row: rows)
        grid.emplace_back(vector<char>(row.begin(), row.end()));
    int res = sol.numIslands(grid);
    if (res == expected)
        cout << "PASS" << endl;
    else
        cout << "FAIL: expect " << expected << " but got " << res << endl;
}

int main() {
    TEST({"11110", "11010", "11000", "00000"}, 1);
    TEST({"11000", "11000", "00100", "00011"}, 3);
}

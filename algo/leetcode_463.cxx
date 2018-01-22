#include <iostream>
#include <vector>

using namespace std;

auto island = vector< vector<int> > {
    {0,1,0,0},
    {1,1,1,0},
    {0,1,0,0},
    {1,1,0,0}
};

int search(vector< vector<int> >& grid, int i, int j, int prev) {
    int m = grid.size(), n = grid[0].size();
    if (i < 0 || i >= m || j < 0 || j >= n)
        return prev;

    switch (grid[i][j]) {
    case -1: return prev - 1; // the edge is encountered twitce
    case 0: return prev;
    default: break;
    }

    // All shared edges are crossed twice
    // With an initial phantom edge
    int curr = prev + 3; // deduct cross from prev
    grid[i][j] = -1;
    for (int di = -1; di <= 1; ++di) {
        for (int dj = -1; dj <= 1; ++dj) {
            if ((0 != di) == (0 != dj)) continue; // ~XOR
            curr = search(grid, i + di, j + dj, curr);
        }
    }
    return curr;
}

int islandPerimeter(vector< vector<int> >& grid) {
    int curr_max = 0;
    int m = grid.size(), n = grid[0].size();
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            if (1 == grid[i][j])
                curr_max = max(curr_max, search(grid, i, j, 1));
    return curr_max;
}


int main() {

    auto isl0 = vector< vector<int> > {
        {0,1,0,0},
        {1,1,1,0},
        {0,1,0,0},
        {1,1,0,0}
    };
    cout << islandPerimeter(isl0) << endl;

    auto isl1 = vector< vector<int> > {{1, 1}};
    cout << islandPerimeter(isl1) << endl;
    
    auto isl2 = vector< vector<int> > {{1,1}, {1,1}};
    cout << islandPerimeter(isl2) << endl;
}

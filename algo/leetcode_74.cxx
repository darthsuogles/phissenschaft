#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

class Solution {
public:
    bool searchMatrix(vector<vector<int>> &matrix, int target) {
        auto n = matrix.size();
        if (0 == n) return false;
        auto m = matrix[0].size();
        if (0 == m) return false;
        if (target < matrix[0][0] || matrix[n-1][m-1] < target)
            return false;

        vector<int> leadings;
        for (auto row: matrix) {
            leadings.push_back(row[0]);
        }
        // Always points to the strictly larger element
        auto one_plus_iter = upper_bound(leadings.begin(), leadings.end(), target);
        int idx = one_plus_iter - leadings.begin();
        if (--idx < 0) return false;

        // Must use "lower_bound" to find if there is a match
        auto cell = lower_bound(matrix[idx].begin(), matrix[idx].end(), target);
        return *cell == target;
    }
};

Solution sol;

void TEST(vector<vector<int>> matrix, int target, const bool expected) {
    auto res = sol.searchMatrix(matrix, target);
    if (res == expected)
        cout << "PASS" << endl;
    else
        cout << "FAIL" << endl;
}

int main() {
    TEST({{1, 3, 5, 7}, {10, 11, 16, 20}, {23, 30, 34, 50}}, 3, true);
    TEST({{1, 3}}, 3, true);
}

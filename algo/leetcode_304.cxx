#include <iostream>
#include <vector>

using namespace std;

class NumMatrix {
private:
    int _num_rows;
    int _num_cols;
    vector< vector<int> > _sub_region;
public:
    NumMatrix(vector< vector<int> > matrix) {
        if (matrix.empty() || matrix[0].empty()) {
            _num_rows = _num_cols = 0;
            return;
        }
        _num_rows = matrix.size();
        _num_cols = matrix[0].size();

        for (int i = 0; i <= _num_rows; ++i) {
            vector<int> row(1 + _num_cols, 0);
            _sub_region.push_back(row);
        }
        cout << _sub_region.size() << endl;

        for (int i = 1; i <= _num_rows; ++i) {
            for (int j = 1; j <= _num_cols; ++j) {
                _sub_region[i][j] = matrix[i-1][j-1]
                    + _sub_region[i-1][j]
                    + _sub_region[i][j-1]
                    - _sub_region[i-1][j-1];
            }
        }
    }

    int sumRegion(int row1, int col1, int row2, int col2) {
        if (row1 > row2 || row2 >= _num_rows)
            return 0;
        if (col1 > col2 || col2 >= _num_cols)
            return 0;
        return _sub_region[row2 + 1][col2 + 1]
            - _sub_region[row1][col2 + 1]
            - _sub_region[row2 + 1][col1]
            + _sub_region[row1][col1];
    }
};

void print_result(NumMatrix &obj, int r1, int c1, int r2, int c2) {
    cout << r1 << ", " << c1 << " => "
         << r2 << ", " << c2 << ": area = "
         << obj.sumRegion(r1, c1, r2, c2) << endl;
}

int main() {
    vector< vector<int> > matrix = {
            {3, 0, 1, 4, 2},
            {5, 6, 3, 2, 1},
            {1, 2, 0, 1, 5},
            {4, 1, 0, 1, 7},
            {1, 0, 3, 0, 5}};

    cout << matrix.size() << " " << matrix[0].size() << endl;
    NumMatrix obj(matrix);
    print_result(obj, 2, 1, 4, 3);
    print_result(obj, 1, 1, 2, 2);
    print_result(obj, 1, 2, 2, 4);
}

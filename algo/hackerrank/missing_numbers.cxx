#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

int main() {
    int n; cin >> n;
    vector<int> A(n);
    for (int i = 0; i < n; ++i) {
        cin >> A[i];
    }
    int m; cin >> m;
    vector<int> B(m);
    for (int j = 0; j < m; ++j) {
        cin >> B[j];
    }
    sort(A.begin(), A.end());
    sort(B.begin(), B.end());

    int i = 0, j = 0;
    for (; i < n; ++j) {
        if (B[j] < A[i]) {
            cout << B[j] << " ";
        } else {
            ++i;
        }
    }
    for (; j < m; ++j) {
        cout << B[j] << " ";
    }
    cout << endl;
    return 0;
}

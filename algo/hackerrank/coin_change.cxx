#include <iostream>
#include <vector>

using namespace std;

void call_it_dp_or_bfs() {
    int n, M;
    cin >> n >> M;
    vector<int> denoms(M);
    for (int i = 0; i < M; cin >> denoms[i++]);
    vector<long> tbl_cnts(n + 1, 0);
    tbl_cnts[0] = 1;
    for (auto denom: denoms) {
        for (int val = denom; val <= n; ++val) {
            tbl_cnts[val] += tbl_cnts[val - denom];
        }
    }
    cout << tbl_cnts[n] << endl;
}

int main() {
    call_it_dp_or_bfs();
}

#include <iostream>
#include <vector>

using namespace std;

int magicalString(int n) {
    if (0 == n) return 0;
    vector<int> arr;
    arr.push_back(1);
    bool is_one = false;
    int res = 1;
    int cnt = 1;
    for (int i = 0; cnt < n; ++i) {
        int a = arr[i];
        if (is_one) {
            switch (a) {
            case 2: arr.push_back(1); if (++cnt == n) break; ++res;
            case 1: arr.push_back(1); if (++cnt == n) break; ++res; break;
            }
        } else {
            switch (a) {
            case 2: arr.push_back(2); if (++cnt == n) break;
            case 1: arr.push_back(2); if (++cnt == n) break; break;
            }
        }
        is_one = !is_one;
    }
    return res;
}

void TEST(int n, int tgt) {
    int res = magicalString(n);
    if (res == tgt) cout << "OK" << endl;
    else cout << "ERR: " << res << " != " << tgt << endl;
}

int main() {
    TEST(6, 3);
    TEST(3, 1);
}

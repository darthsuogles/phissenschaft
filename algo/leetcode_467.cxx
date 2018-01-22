#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;

int findSubstringInWraproundString(string p) {
    int res = 0;
    int n = 1;    
    p = p + "#";
    int tbl[26] = {0};  // length of substr till char_idx
    for (int i = 0; i + 1 < p.size(); ++i) {
        int idx_curr = p[i] - 'a';
        if ((1 + idx_curr) % 26 == (p[i+1] - 'a')) {
            ++n;
        } else {
            n = 1;
        }
        if (n > tbl[idx_curr]) { // update at every iteration
            res += n - tbl[idx_curr];  // only update extra
            tbl[idx_curr] = n;
        }
    }
    return res;
}

#define TEST(str, val) {                                    \
        int res = findSubstringInWraproundString((str));    \
        if (res == (val)) cout << "OK" << endl;             \
        else cout << "ERR: " << res << endl; }

int main() {
    TEST("a", 1);
    TEST("cac", 2);
    TEST("zab", 6);
    TEST("aabb", 3);
    TEST("zaba", 6);
}

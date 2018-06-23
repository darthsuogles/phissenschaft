#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> lexicalOrder(int n) {
        vector<int> res(n);
        int cur = 1;
        for (int i = 0; i < n; ++i) {
            res[i] = cur;
            if (cur * 10 <= n) {
                cur *= 10;
                continue;
            }
            if (cur >= n) cur /= 10;
            ++cur;
            while (0 == cur % 10) cur /= 10;
        }
        return res;
    }

    void lexDFS(int prefix, int d, int n, vector<int> &res) {
        int val = prefix * 10 + d;
        if (val > n) return;
        res.push_back(val);
        for (int i = 0; i < 10; ++i) {
            lexDFS(val, i, n, res);
        }
    }

    vector<int> lexicalOrderWithDFS(int n) {
        vector<int> res;
        for (int d = 1; d < 10; ++d) {
            lexDFS(0, d, n, res);
        }
        return res;
    }

    vector<int> lexicalOrderWithSort(int n) {
        vector<int> vals(n);
        for (int d = 1; d <= n; ++d) vals[d - 1] = d;
        sort(vals.begin(), vals.end(), [](int a, int b) {
                int a_rev = 0, b_rev = 0;
                int a_len = 0, b_len = 0;
                for (; a > 0; ++a_len) { a_rev = a_rev * 10 + (a % 10); a /= 10; }
                for (; b > 0; ++b_len) { b_rev = b_rev * 10 + (b % 10); b /= 10; }
                while (a_rev > 0 && b_rev > 0) {
                    int a_rmd = a_rev % 10;
                    int b_rmd = b_rev % 10;
                    if (a_rmd == b_rmd) {
                        a_rev /= 10;
                        b_rev /= 10;
                        continue;
                    }
                    return a_rmd < b_rmd;
                }
                if (a_rev == b_rev) return a_len < b_len;
                return a_rev < b_rev;
            });
        return vals;
    }
};

Solution sol;

int TEST(int n) {
    auto res_A = sol.lexicalOrder(n);
    auto res_B = sol.lexicalOrderWithSort(n);
    if (res_A.size() != res_B.size()) {
        cerr << "FAIL: non-equal len: " << res_A.size() << " vs " << res_B.size() << endl;
        return 1;
    }
    for (int i = 0; i < res_A.size(); ++i) {
        if (res_A[i] != res_B[i]) {
            cerr << "FAIL: at " << i << ": " << res_A[i] << " != " << res_B[i] << endl;
            return 1;
        }
    }
    cout << "PASS" << endl;
    return 0;

}

int main() {
    TEST(13);
    TEST(10);
    TEST(8);
    TEST(45);
    TEST(112);
    TEST(3232);
}

#include <iostream>
#include <string>

using namespace std;

int main() {
    int N; cin >> N;
    string s; cin >> s;

    int max_cnts = 0;
    int tbl[26] = {0};
    for (auto ch: s) ++tbl[ch - 'a'];
    for (int i = 0; i < 25; ++i) {
        if (0 == tbl[i]) continue;
        for (int j = i + 1; j < 26; ++j) {
            if (0 == tbl[j]) continue;
            char ci = i + 'a', cj = j + 'a';

            // Look for the first occurrence
            int k = 0;
            for (; k < N; ++k)
                if (ci == s[k] || cj == s[k]) break;
            if (k == N) continue;

            int cnts = 1; // found one already
            int idx = k++; // skip the last found char
            for (; k < N; ++k) {
                if (ci != s[k] && cj != s[k]) continue;
                if (s[idx] == s[k]) break;
                ++cnts;
                idx = k;
            }
            if (k != N) continue;
            max_cnts = max(cnts, max_cnts);
        }
    }
    cout << max_cnts << endl;
}

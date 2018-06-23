#include <iostream>
#include <string>

using namespace std;

int strStr(string haystack, string needle) {
    if (needle.empty()) return 0;
    int m = haystack.size(), n = needle.size();
    if (m < n) return -1;

    using integer = unsigned long long;
    const integer MOD = 1e9 + 7;
    const integer BASE = 127;

    integer roll_val = 0;
    integer sig_val = 0;
    int i = 0;
    for (; i < n; ++i) {
        roll_val = (roll_val * BASE) % MOD;
        roll_val = (roll_val + static_cast<integer>(haystack[i])) % MOD;
        sig_val = (sig_val * BASE) % MOD;
        sig_val = (sig_val + static_cast<integer>(needle[i])) % MOD;
    }
    for (; i <= m; ++i) {
        if (sig_val == roll_val) {
            int j = i - n;
            for (int k = 0; j < i; ++j, ++k) {
                if (haystack[j] != needle[k]) break;
            }
            if (j == i) return i - n;
        }
        if (m == i) return -1;
        auto sub_val = static_cast<integer>(haystack[i - n]);
        for (int k = 1; k < n; ++k) sub_val = (sub_val * BASE) % MOD;
        roll_val = (roll_val + MOD - sub_val) % MOD;
        roll_val = (roll_val * BASE) % MOD;
        roll_val = (roll_val + static_cast<integer>(haystack[i])) % MOD;
    }
    return -1;
}

void TEST(string a, string b) {
    auto shift = strStr(a, b);
    if (-1 == shift) {
        cout << "NO MATCH" << endl;
    } else {
        cout << a << endl;
        for (int i = 0; i < shift; ++i) cout << " ";
        cout << b << endl;
    }
    cout << "=============" << endl;
}

int main() {
    TEST("hello", "ll");
    TEST("hello", "llo");
}

#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;

bool canConstruct(string ransomNote, string magazine) {
    unordered_map<char, int> tbl;
    for (auto ch : ransomNote) ++tbl[ch];
    for (auto ch : magazine) --tbl[ch];
    for (auto& kv : tbl) {
        if (kv.second > 0) return false;
    }
    return true;
}

#define TEST(s1, s2, ans) {\
        if (canConstruct((s1), (s2)) == ans) cout << "OK" << endl; \
        else cout << "ERR" << endl; }

int main() {
    TEST("a", "b", false);
    TEST("aa", "ab", false);
    TEST("aa", "aab", true);
}

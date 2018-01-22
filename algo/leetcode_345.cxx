#include <iostream>
#include <string>

using namespace std;

#pragma inline
bool is_vowel(char ch) { 
    switch (ch) {
    case 'a': case 'e': case 'i': case 'o': case 'u': 
    case 'A': case 'E': case 'I': case 'O': case 'U': 
        return true;
    default: return false;
    }
}

string reverseVowels(string s) {
    if (s.empty()) return s;
    auto n = s.size();
    for (int i = 0, j = n - 1; i < j; ++i, --j) {
        for (; i < n; ++i) if (is_vowel(s[i])) break;
        for (; j >= 0; --j) if (is_vowel(s[j])) break;
        if (i >= j) break;
        char tmp = s[i]; s[i] = s[j]; s[j] = tmp;        
    }
    return s;
}

void TEST(string s, string tgt) {
    auto res = reverseVowels(s);
    if (res == tgt)
        cout << "OK";
    else 
        cout << "ERROR: " << res << " != " << tgt;
    cout << endl;
}

int main() {
    TEST("hello", "holle");
    TEST("leetcode", "leotcede");
    TEST(" ", " ");
    TEST("OE", "EO");
    TEST(".,", ".,");
}

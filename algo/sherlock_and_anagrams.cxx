#include <iostream>
#include <string>
#include <unordered_map>
#include <algorithm>

using namespace std;

using integer = long;

integer find_num_anagrams(string s) {
    auto n = s.size();
    if (0 == n) return 0;

    integer num_anagrams = 0;
    for (int sz = 1; sz < n; ++sz) {
        unordered_map<string, integer> cache;
        for (int i = 0; i + sz <= n; ++i) {
            int j = i + sz;
            auto sub = s.substr(i, sz);
            sort(sub.begin(), sub.end());
            auto cnts = cache[sub];
            if (cnts > 0) {
                num_anagrams += cnts;
            }
            ++cache[sub];
        }
    }
    return num_anagrams;
}

int main() {
    int N; cin >> N;
    for (int i = 0; i < N; ++i) {
        string line; cin >> line;
        cout << find_num_anagrams(line) << endl;
    }
}

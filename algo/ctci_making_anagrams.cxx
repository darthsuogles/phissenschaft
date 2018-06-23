#include <iostream>
#include <vector>

using namespace std;

#define ABS(a) (a) < 0 ? -(a) : (a)

int main() {
    string s1, s2;
    cin >> s1;
    cin >> s2;

    int histo[26] = {0};
    for (auto ch: s1) ++histo[ch - 'a'];
    for (auto ch: s2) --histo[ch - 'a'];
    int cnts = 0;
    for (int i = 0; i < 26; ++i) {
        cnts += ABS(histo[i]);
    }
    cout << cnts << endl;
}

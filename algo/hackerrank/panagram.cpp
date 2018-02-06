#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <ctype.h>

using namespace std;


int main() {
    /* Enter your code here. Read input from STDIN. Print output to STDOUT */   
    bool tbl_char_used[26];
    for (int d = 0; d < 26; tbl_char_used[d++] = false);
    string S; getline(cin, S);

    int uniq_char_cnt = 0;
    bool is_panagram = false;
    for (int i = 0; i < S.size(); ++i) {
        char ch = tolower(S[i]);
        if ( ch < 'a' || ch > 'z' ) continue;
        int ch_idx = ch - 'a';
        if ( ! tbl_char_used[ch_idx] ) {
            tbl_char_used[ch_idx] = true;        
            if ( 26 == ++uniq_char_cnt ) {
                is_panagram = true;
                break;   
            }
        }
    }
    if ( ! is_panagram ) cout << "not ";
    cout << "pangram";
    return 0;
}


#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

int main() {
    string s_curr;
    cin >> s_curr;
    string s_next;

    while (true) {
        s_curr.push_back('$');
        auto len = s_curr.size();
        bool reduced = false;
        for (int i = 0; i + 1 < len; ++i) {
            if (s_curr[i] == s_curr[i+1]) {
                ++i; reduced = true;
            } else s_next.push_back(s_curr[i]);
        }
        s_curr = s_next; s_next.clear();
        if (!reduced) break;
    }
    if (s_curr.empty())
        cout << "Empty String" << endl;
    else
        cout << s_curr << endl;
    return 0;
}

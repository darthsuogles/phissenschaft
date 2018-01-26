#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int findContentChildren(vector<int>& g, vector<int>& s) {
    sort(g.begin(), g.end(), greater<int>());
    sort(s.begin(), s.end(), greater<int>());
    int res = 0;
    for (int i = 0, j = 0; i < g.size() && j < s.size(); ++i) {
        if (g[i] > s[j]) continue;
        ++res; ++j;
    }
    return res;
}

void TEST(vector<int> g, vector<int> s, int tgt) {
    int res = findContentChildren(g, s);
    if (tgt == res) cout << "OK" << endl;
    else cout << "ERR " << res << " != " << tgt << endl;
}

int main() {
    TEST({1,2}, {1,2,3}, 2);
    TEST({1,2,3}, {1,1}, 1);
}

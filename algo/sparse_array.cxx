/**
 * Check the presense of queries in a string collection
 */

#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;

int main() {
    int N; cin >> N;
    unordered_map<string, int> str_coll;
    string line;
    for (int i = 0; i < N; ++i) {
        cin >> line;
        ++str_coll[line];
    }
    int Q; cin >> Q;
    for (int i = 0; i < Q; ++i) {
        cin >> line;
        cout << str_coll[line] << endl;
    }
}

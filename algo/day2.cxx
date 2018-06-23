#include <iostream>
#include <sstream>
#include <utility>
#include <climits>

using namespace std;

int main() {
    string line;
    long sum = 0;
    while (getline(cin, line)) {
        stringstream sstrm(line);
        int min_val = INT_MAX;
        int max_val = INT_MIN;
        int val;
        while (sstrm >> val) {
            min_val = min(val, min_val);
            max_val = max(val, max_val);
        }
        sum += max_val - min_val;
    }
    cout << sum << endl;
}

#include <iostream>

using namespace std;

bool canWinNimLoop(int n) {
    bool p1 = true, p2 = true, p3 = true, res = true;
    for (int i = 3; i < n; ++i) {
        res = !p1 || !p2 || !p3;
        p3 = p2; p2 = p1; p1 = res;
    }
    return res;
}

bool canWinNim(int n) {
    if (n <= 3) return true;
    return 0 != (n & 3);  // consecutive 3 wins
}

int main() {
    cout << canWinNim(3) << endl;
    cout << canWinNim(4) << endl;
    cout << canWinNim(7) << endl;
}

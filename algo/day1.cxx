#include <iostream>

using namespace std;

int main() {
    char init_digit; cin >> init_digit;
    char prev = init_digit;
    char curr;
    long sum = 0;
    while (cin >> curr) {
        if (curr == prev) {
            sum += static_cast<int>(curr - '0');
        }
        prev = curr;
    }
    if (curr == init_digit)
        sum += static_cast<int>(curr - '0');
    cout << sum << endl;
}

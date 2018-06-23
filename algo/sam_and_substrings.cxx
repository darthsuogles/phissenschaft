#include <iostream>
#include <vector>
#include <deque>

using namespace std;

int main() {
    using integer = unsigned long long;
    char ch;
    const integer MOD = 10e9 + 7;
    cin >> ch;
    integer last_digit = static_cast<integer>(ch - '0');
    integer prev = last_digit;
    integer total = prev;
    integer prefix_len = 1;
    while (cin >> ch) {
        integer d = static_cast<integer>(ch - '0');
        // We still would count 05 and 5 as two instances of 5
        // Thus the following line is not used
        //auto d_mult = prefix_len - static_cast<integer>(0 == last_digit);
        last_digit = d;
        auto prev_incr = (prev * 10) % MOD + (d * prefix_len) % MOD;
        auto curr = (prev_incr + d) % MOD;
        total = (total + curr) % MOD;
        prev = curr;
        ++prefix_len;
    }
    cout << total << endl;
}

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>

using namespace std;

class Solution {
public:
    string fractionToDecimal(int numerator, int denominator) {
        return aux(static_cast<long long>(numerator),
                   static_cast<long long>(denominator));
    }
    string aux(long long numerator, long long denominator) {
        if (0 == denominator) return "";
        if (0 == numerator) return "0";
        if (numerator < 0 || denominator < 0) {
            auto prefix = ((numerator < 0) != (denominator < 0)) ? "-" : "";
            return prefix + aux(abs(numerator), abs(denominator));
        }

        long long int_part = numerator / denominator;
        long long a = numerator % denominator;
        unordered_map<long long, int> prev_index_of;
        string partial;
        for (int i = 0; a != 0; ++i) {
            if (prev_index_of.count(a) > 0) {
                int idx = prev_index_of[a];
                partial = partial.substr(0, idx) +
                    "(" + partial.substr(idx) + ")";
                break;
            }
            prev_index_of[a] = i;
            a *= 10;
            long long digit = (a < denominator) ? 0 : (a / denominator);
            partial += to_string(digit);
            a = a % denominator;
        }

        if (partial.empty()) return to_string(int_part);
        return to_string(int_part) + "." + partial;
    }
};

Solution sol;

void TEST(int num, int denom) {
    cout << sol.fractionToDecimal(num, denom) << endl;
}

int main() {
    TEST(1, 2);
    TEST(2, 1);
    TEST(2, 3);
    TEST(1, 6);
    TEST(1, 7);
    TEST(-22, -2);
    TEST(-1, -2147483648);
    TEST(1, 333);
    TEST(1, 90);
    TEST(0, -5);
}

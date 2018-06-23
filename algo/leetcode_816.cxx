#include <iostream>
#include <string>
#include <vector>

using namespace std;

class Solution {
    vector<string> parseOneNumber(string expr) {
        vector<string> res;
        if (expr.empty()) return res;
        auto len = expr.size();
        // If the number starts with zero, must be decimal
        if ('0' != expr[0] || 1 == len) res.push_back(expr);
        // Trailing zeros are accepted for non-decimal
        if ('0' == expr.back()) return res;
        if ('0' == expr[0]) {
            auto new_expr = expr;
            new_expr.insert(1, 1, '.');
            res.push_back(new_expr);
            return res;
        }
        for (int i = 1; i < len; ++i) {
            auto new_expr = expr;
            new_expr.insert(i, 1, '.');
            res.push_back(new_expr);
        }
        return res;
    }

public:
    vector<string> ambiguousCoordinates(string S) {
        vector<string> res;
        if (S.size() < 2) return res;
        if (S[0] != '(' || S.back() != ')') return res;
        S = S.substr(1, S.size() - 2);
        auto len = S.size();
        if (len < 2) return res;
        for (int prev_len = 1; prev_len < len; ++prev_len) {
            auto prev_nums = parseOneNumber(S.substr(0, prev_len));
            if (prev_nums.empty()) continue;
            auto post_nums = parseOneNumber(S.substr(prev_len));
            for (auto prev_num: prev_nums) {
                for (auto post_num: post_nums) {
                    res.push_back("(" + prev_num + ", " + post_num + ")");
                }
            }
        }
        return res;
    }
};

Solution sol;

void TEST(string expr) {
    cout << "==================================" << endl;
    for (auto coords: sol.ambiguousCoordinates(expr)) {
        cout << coords << endl;
    }
}

int main() {
    TEST("(123)");
    TEST("(00011)");
    TEST("(0123)");
    TEST("(100)");
}

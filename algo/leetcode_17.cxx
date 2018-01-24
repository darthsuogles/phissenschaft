#include <iostream>
#include <string>
#include <vector>

using namespace std;

class Solution {
    string get_combo(char ch) {
        int digit = int(ch - '0');
        switch (digit) {
        case 2: return "abc";
        case 3: return "def";
        case 4: return "ghi";
        case 5: return "jkl";
        case 6: return "mno";
        case 7: return "pqrs";
        case 8: return "tuv";
        case 9: return "wxyz";
        default: return "";
        }
    }

public:
    vector<string> letterCombinations(string digits) {
        if (digits.empty()) return {};
        vector<string> *res_curr, *res_next;
        res_curr = new vector<string>(1, "");
        res_next = new vector<string>();
        for (char d: digits) {
            for (char ch: get_combo(d)) {
                for (auto expr: *res_curr)
                    res_next->push_back(expr + ch);
            }
            swap(res_curr, res_next);
            res_next->clear();
        }
        return *res_curr;
    }
};

Solution sol;
void TEST(string digits) {
    for (auto expr: sol.letterCombinations(digits))
        cout << expr << " ";
    cout << endl;
}

int main() {
    TEST("23");
}

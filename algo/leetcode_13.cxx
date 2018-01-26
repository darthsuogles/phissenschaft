#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <cassert>

using namespace std;

/**
 * The easiest way is to work from right to left
 * If the next signicicant literal is smaller,
 * we modify the current value as minus, 
 * otherwise, add that number
 *
 * Ref: https://www.rosettacode.org/wiki/Roman_numerals
 */
int romanToInt(string s) {
    unordered_map<char, int> romlit2int = {
        {'I', 1},
        {'V', 5},
        {'X', 10},
        {'L', 50},
        {'C', 100},
        {'D', 500},
        {'M', 1000}
    };
    
    int tot = 0, curr = 0, prev = 0;
    for (auto ich = s.rbegin(); ich != s.rend(); ++ich) {
        curr = romlit2int[*ich];
        tot += curr < prev ? -curr : curr;
        prev = curr;
    }
    return tot;
}

string intToRoman(int num) {
    struct roman_lit_t { int val; char const *lit; };
    static roman_lit_t const roman_data[] = {
        1000, "M",
        900, "CM",
        500, "D",
        400, "CD",
        100, "C",
        90, "XC",
        50, "L",
        40, "XL",
        10, "X",
        9, "IX",
        5, "V",
        4, "IV",
        1, "I",
        0, NULL
    };
    string roman;
    for (roman_lit_t const *curr = roman_data;
         curr->lit != NULL; ++curr) {
        while (num >= curr->val) {
            roman += curr->lit;
            num -= curr->val;
        }
        if (0 == num) break;
    }
    return roman;
}

void TEST(string orig, int ref) {
    int val = romanToInt(orig);
    string roman = intToRoman(ref);
    cout << orig << " <=> " << roman << " || " << ref << " <=> " << val << endl;
    assert( val == ref && orig == roman );
}

int main() {
    TEST("DCXXI", 621);
    TEST("MCMLIV", 1954);
    TEST("MCMXC", 1990);
    TEST("MMXIV", 2014);
}

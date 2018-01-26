/**
 * LeetCode Problem 10: simple regular expression match
 *
 * '.' Matches any single character.
 * '*' Matches zero or more of the preceding element.
 *
 * The matching should cover the entire input string (not partial).
 * https://leetcode.com/problems/regular-expression-matching/solution/
 */

#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <cstring>
#include <cassert>

using namespace std;

class Solution {
    unordered_map<size_t, bool> tbl_matched_;

public:
    bool isMatch(string s, string p) {
        //return isMatch(s.c_str(), 0, p.c_str(), 0);
        return isMatchRec(s.c_str(), p.c_str());
    }

    // Simple recursive method
    bool isMatchRec(const char *s, const char *p) {
        assert(nullptr != s && nullptr != p);
        if (p[0] == '\0') return s[0] == '\0';
        assert(p[0] != '*');

        if (p[1] != '*') {
            if (s[0] == '\0') return false;
            if (s[0] == p[0] || '.' == p[0])
                return isMatchRec(s + 1, p + 1);
        } else { // Kleen star
            bool skip_star_match = isMatchRec(s, p + 2);
            if (skip_star_match) return true;
            if (s[0] == '\0') return false;
            if (s[0] == p[0] || '.' == p[0])
                return isMatchRec(s + 1, p);
        }
        return false;
    }

    bool isMatch(const char *s0, int i, const char *p0, int j) {
        assert( p0 != NULL && s0 != NULL );
        const char *s = s0 + i;
        const char *p = p0 + j;
        assert( p != NULL && s != NULL );
        if ( p[0] == '\0' )
            return ( s[0] == '\0' );

        if ( p[0] == '*' ) // the first one cannot be a Kleen star
            return false;

        // Check cache
        auto key = (size_t) i << 32 | (unsigned int) j;
        auto vit = tbl_matched_.find(key);
        if ( vit != tbl_matched_.end() ) {
            return vit->second;
        }
#define YIELD(EXPR) return tbl_matched_[key] = (EXPR)

        if ( p[1] == '*' ) {
            bool is_match = isMatch(s0, i, p0, j + 2); // match a* to empty
            if ( is_match )
                YIELD(true);
            if ( s[0] == '\0' )
                YIELD(false);
            if ( p[0] == '.' || s[0] == p[0] ) // match one character
                YIELD(isMatch(s0, i + 1, p0, j));
        } else {
            if ( s[0] == '\0')
                YIELD(false);
            if ( p[0] == '.' || s[0] == p[0] )
                YIELD(isMatch(s0, i + 1, p0, j + 1));
        }
        YIELD(false);
#undef YIELD
    }
};

void test(Solution sol, string s, string p, bool is_match) {
    cout << s << "\t:\t" << p << endl;
    assert(sol.isMatch(s, p) == is_match);
}


int main() {
    Solution sol;
#define TEST(T, P, R) test(sol, (T), (P), (R))

    TEST("aa", "a", false);
    TEST("aa", "aa", true);
    TEST("aaa", "a*", true);
    TEST("ab", ".*", true);
    TEST("aaaa", "a", false);
    TEST("aab", "c*a*b", true);
    TEST("ab", ".*c", false);
    TEST("aaa", "aaaa", false);
    TEST("aaa", "a*a", true);
    TEST("aaaaaaaaaaaaab", "a*a*a*a*a*a*a*a*a*a*c", false);

    TEST("a", ".*..a*", false);

    return 0;
}

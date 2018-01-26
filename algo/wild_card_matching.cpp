#include <iostream>

using namespace std;

bool isMatchAux(const string &text, int i,
                const string &patt, int j) {
    if ( text.empty() || text.size() == i ) {
        for (int k = j; k < patt.size(); ++k) {
            if ( '*' != patt[k] ) return false;
        }
        return true;
    }
    if ( patt.empty() || patt.size() == j )
        return false;

    char ch = text[i], p = patt[j];
    if ( ch == p || '?' == p ) {
        return isMatchAux(text, i+1, patt, j+1);
    }
    if ( '*' != p )
        return false;

    for (int k = i; k <= text.size(); ++k) {
        if ( isMatchAux(text, k, patt, j+1) )
            return true;
    }
    return false;
}

bool isMatch(string text, string patt) {
    return isMatchAux(text, 0, patt, 0);
}

/**
 * Without recursive call
 */
bool isMatchIter(string text, string patt) {
    int m = text.size();
    int n = patt.size();

    /**
     * We only have to track one set of (i, j) for '*'
     * 1. Consecutive '*' are effectively just one '*'
     * 2. For non-consecutive '*', any non greedy match of
     *    the first '*' will have an equivalent second '*' matching.
     */
    int btrcPosI = -1;
    int btrcPosJ = -1;
    int i = 0; // text iterator
    int j = 0; // pattern iterator
    while ( i < m ) {
        if ( j < n ) {
            char chText = text[i];
            char chPatt = patt[j];
            if ( '?' == chPatt || chPatt == chText ) {
                ++i; ++j;
                continue;
            }
            if ( '*' == chPatt ) {
                btrcPosI = i;
                for (; j < n; ++j)
                    if ( patt[j] != '*' ) break;
                if ( j == n ) return true;
                btrcPosJ = j;
                continue;
            }
        }
        // Either non-match or j == n
        if ( -1 == btrcPosI )
            return false;
        i = ++btrcPosI;
        j = btrcPosJ;
    }
    for (; j < n; ++j) // the rest in pattern must all be '*'
        if ( patt[j] != '*' ) return false;
    return true;
}

void testCase(string text, string patt, bool res) {
    cout << text << endl;
    cout << patt << endl;
    cout << "Test Passed ?: "
         << boolalpha << (res == isMatchIter(text, patt)) << endl;
    cout << noboolalpha << "----------------------" << endl;
}

int main() {
    testCase("aa", "*", true);
    testCase("aa", "aa", true);
    testCase("aa", "a", false);
    testCase("a", "?", true);
    testCase("aab", "c*a*b", false); // * not follow the previous char
    testCase("cddab", "**c***a*b*", true);
    testCase("a", "", false);
    testCase("", "*", true);
    testCase("", "a", false);
}

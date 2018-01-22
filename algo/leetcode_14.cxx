#include <iostream>
#include <string>
#include <vector>

using namespace std;

class Solution {
private:

    string findLongest(vector<string> &strs, int len) {
        string res;
        if (0 == len) return res;
        for (int i = 0; i < len; ++i) {
            char ch = strs[0][i];
            for (auto s: strs) {
                if (s[i] != ch)
                    return res;
            }
            res += ch;
        }
        return res;
    }

public:

#define MIN(X, Y) (X) < (Y) ? (X) : (Y)

    // Use binary search
    string longestCommonPrefix(vector<string> &strs) {
        string res;
        if (strs.empty()) return res;
        int minLen = 1e9;
        for (auto s: strs)
            minLen = MIN(minLen, s.size());

        size_t i = 0, j = minLen + 1;
        while (i < j) {
            size_t k = (i + j) / 2;
            res = findLongest(strs, k);
            if (res.size() < k)
                j = k;
            else
                i = k + 1;
        }
        return strs[0].substr(0, j - 1);
    }

    string longestCommonPrefixVertScan(vector<string> &strs) {
        string res;
        if (strs.empty()) return res;
        for (int i = 0; ; ++i) {
            if (i >= strs[0].size())
                break;
            char ch = strs[0][i];
            for (auto s: strs) {
                if (i >= s.size() || s[i] != ch)
                    return res;
            }
            res += ch;
        }
        return res;
    }
};

int main() {
    Solution sol;
    vector<string> strs = {"aaa", "aab", "aac"};
    cout << sol.longestCommonPrefix(strs) << endl;
    vector<string> strs_empty = {};
    cout << sol.longestCommonPrefix(strs_empty) << endl;
}

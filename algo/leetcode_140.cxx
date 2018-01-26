#include <iostream>
#include <string>
#include <algorithm>
#include <vector>

using namespace std;

class TrieNode {
private:
    bool is_terminal_;
    TrieNode* children_[26];

    TrieNode* insert(char ch) {
        int idx = int(ch - 'a');
        if (nullptr == children_[idx]) {
            children_[idx] = new TrieNode();
        }
        return children_[idx];
    }

public:
    TrieNode() {
        for (int i = 0; i < 26; children_[i++] = nullptr);
        is_terminal_ = false;
    }

    const bool is_terminal() { return is_terminal_; }

    TrieNode* next(char ch) {
        return children_[int(ch - 'a')];
    }

    TrieNode* insert(string s) {
        TrieNode *node = this;
        for (char ch: s) {
            node = node->insert(ch);
        }
        node->is_terminal_ = true;
        return node;
    }
};

class Solution {
    vector<string> add_matches(int idx,
                               vector<vector<int>> &matched_prefix_inds,
                               string &word) {
        vector<string> res;
        if (idx < 0) return res;
        if (matched_prefix_inds[idx].empty())
            return res;
        for (int i: matched_prefix_inds[idx]) {
            string suffix = word.substr(i + 1, idx - i);
            if (-1 == i) {
                res.push_back(suffix);
                continue;
            }
            auto sub_matches = add_matches(i, matched_prefix_inds, word);
            for (auto prefix: sub_matches) {
                res.push_back(prefix + " " + suffix);
            }
        }
        return res;
    }

public:
    vector<string> wordBreak(string s, vector<string> &wordDict) {
        TrieNode *root = new TrieNode();
        for (string word: wordDict) {
            reverse(word.begin(), word.end());
            root->insert(word);
        }

        vector<bool> tbl_match(s.size() + 1, false);
        tbl_match[0] = true;

        // Stores the matching prefix indices
        vector<vector<int>> matched_prefix_inds(s.size());

        for (int i = 0; i < s.size(); ++i) {
            TrieNode *node = root->next(s[i]);
            for (int j = i - 1; j >= -1 && node; --j) {
                if (node->is_terminal() && tbl_match[j+1]) {
                    matched_prefix_inds[i].push_back(j);
                    tbl_match[i+1] = true;
                }
                node = node->next(s[j]);
            }
        }

        bool is_matched = tbl_match[s.size()];
        vector<string> res;
        if (!is_matched) return res;
        return add_matches(s.size() - 1, matched_prefix_inds, s);
    }
};

void test(Solution &sol, string word, vector<string> dict) {
    cout << "------ TEST --------" << endl;
    auto res = sol.wordBreak(word, dict);
    cout << "word: " << word << endl;
    for (auto word: dict)
        cout << "\t" << word << endl;
    for (auto sent: res)
        cout << sent << endl;
}

int main() {
    Solution sol;
#define TEST(W, ...) test(sol, (W), {__VA_ARGS__})

    TEST("catsanddog", {"cat", "cats", "and", "sand", "dog"});
}

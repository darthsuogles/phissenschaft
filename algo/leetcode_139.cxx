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
public:

    bool wordBreak(string s, vector<string> &wordDict) {
        TrieNode *root = new TrieNode();
        for (string word: wordDict) {
            root->insert(word);
        }

        reverse(s.begin(), s.end());
        vector<bool> tbl_match(s.size() + 1, false);
        tbl_match[0] = true;

        for (int i = 0; i < s.size(); ++i) {
            TrieNode *node = root->next(s[i]);
            for (int j = i - 1; j >= -1 && node; --j) {
                if (node->is_terminal() && tbl_match[j+1]) {
                    tbl_match[i+1] = true;
                    break;
                }
                node = node->next(s[j]);
            }
        }
        return tbl_match[s.size()];
    }
};

int main() {
    Solution sol;

    vector<string> dict = {"abc", "def"};
    cout << sol.wordBreak("abcdef", dict) << endl;

    string wordA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabaabaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    vector<string> dictA = {
        "aa","aaa","aaaa","aaaaa","aaaaaa","aaaaaaa","aaaaaaaa","aaaaaaaaa","aaaaaaaaaa","ba"
    };
    cout << sol.wordBreak(wordA, dictA) << endl;
}

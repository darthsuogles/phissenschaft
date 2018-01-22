/**
 * LeetCode Problem 212: Word Search II
 * 
 * https://leetcode.com/problems/word-search-ii/
 */

#include <iostream>
#include <vector>
#include <unordered_set>
#include <string>

using namespace std;

class TrieNode {
    TrieNode* _children[26];
    bool _is_terminal;
public:    
    TrieNode() {
        for (int i = 0; i < 26; ++i) 
            _children[i] = nullptr;
        _is_terminal = false;
    }
    
    void insert(string word, int idx = 0) {
        if (word.size() == idx) {
            _is_terminal = true; return;
        }
        int i = word[idx] - 'a';
        if (nullptr == _children[i])
            _children[i] = new TrieNode();
        _children[i]->insert(word, idx + 1);
    }

    const bool is_terminal() {
        return _is_terminal;
    }
    
    TrieNode* next(char ch) {
        return _children[ch - 'a'];
    }
};

class Solution {
private:
    void find(TrieNode *root, string &partial, vector<string> &matched,
              int i, int j, vector< vector<char> > &board) {
        if (nullptr == root) return;
        if ('@' == board[i][j]) return;

        int m = board.size(), n = board[0].size();    
        char ch = board[i][j];        
        auto next = root->next(ch);
        if (nullptr == next) return;

        partial.push_back(ch);
        if (next->is_terminal())
            matched.push_back(partial);            

        board[i][j] = '@';
        if ( i-1 >= 0 ) {        
            find(next, partial, matched, i-1, j, board);
        }
        if ( i+1 < m ) {
            find(next, partial, matched, i+1, j, board);
        }
        if ( j-1 >= 0 ) {
            find(next, partial, matched, i, j-1, board);
        }
        if ( j+1 < n ) {
            find(next, partial, matched, i, j+1, board);
        }
        board[i][j] = ch;
        partial.pop_back();
    }
  
public:
    vector<string> findWords(vector< vector<char> >& board, vector<string>& words) {
        if ( board.empty() ) return {};
        int m = board.size();
        if ( board[0].empty() ) return {};
        int n = board[0].size();

        vector<string> res;
        // Try to find each word in dictionary
        auto word_set = unordered_set<string>(words.begin(), words.end());
        TrieNode *root = new TrieNode();
        for (auto w: word_set) 
            root->insert(w);

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {                
                if (word_set.empty()) 
                    return res;
                vector<string> matched;
                string partial;
                find(root, partial, matched, i, j, board);
                for (auto w: matched) {
                    if (word_set.erase(w))
                        res.push_back(w);
                }
            }
        }
        return res;
    }
};

void TEST(vector<string> words, 
          vector< vector<char> > board,
          vector<string> targets) {
    cout << "----------------------" << endl;
    Solution sol;
    auto matched_words = sol.findWords(board, words);
    if (targets.size() != matched_words.size()) {
        cout << "ERROR" << " size unmatched: " << matched_words.size() << " but expect " << targets.size() << endl;
        return;
    }
    sort(matched_words.begin(), matched_words.end());
    sort(targets.begin(), targets.end());
    for (int i = 0; i < matched_words.size(); ++i) {
        auto w1 = matched_words[i]; 
        auto w2 = targets[i];
        if (w1 == w2) {
            cout << w2 << endl; continue;
        }
        cout << "ERROR: mismatched words " << w1 << " but expect " << w2 << endl;
    }
}

    int main() {
        TEST({"oath","pea","eat","rain"}, {
                {'o','a','a','n'},{'e','t','a','e'}, {'i','h','k','r'}, {'i','f','l','v'}
            }, {"oath", "eat"});
        TEST({"a"}, {{'a'}}, {"a"});
        TEST({"a"}, {{'a', 'a'}, {'a', 'b'}}, {"a"});
        TEST({"a", "a"}, {{'a', 'a'}, {'a', 'b'}}, {"a"});
    }

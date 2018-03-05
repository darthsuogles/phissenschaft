#include <iostream>
#include <utility>
#include <memory>
#include <vector>

using namespace std;

class TrieNode: public enable_shared_from_this<TrieNode> {
    // Lower case English letters only
    static const int VOCAB_ = 26;
    vector<shared_ptr<TrieNode>> children_;
    bool is_terminal_;
    // How many node in its subtree (inclusive) are terminal nodes
    int num_terminals_;

public:
    TrieNode(bool is_terminal = false)
        : children_(VOCAB_, nullptr), is_terminal_(is_terminal), num_terminals_(0) {}

    bool is_terminal() {
        return is_terminal_;
    }

    void insert(string s) {
        if (s.empty()) return;
        auto node = shared_from_this();
        for (auto ch: s) {
            int idx = static_cast<int>(ch - 'a');
            if (nullptr == node->children_[idx]) {
                node->children_[idx] = make_shared<TrieNode>();
            }
            ++node->num_terminals_;
            node = node->children_[idx];
        }
        ++node->num_terminals_;
        node->is_terminal_ = true;
    }

    int count_terminals() {
        return num_terminals_;
    }

    int find_partial(string s) {
        if (s.empty()) return 0;
        auto node = shared_from_this();
        for (auto ch: s) {
            node = node->children_[ch - 'a'];
            if (nullptr == node) return 0;
        }
        return node->count_terminals();
    }
};

int main() {
    int n; cin >> n;
    auto root = make_shared<TrieNode>();
    for (int i = 0; i < n; ++i) {
        string op, text;
        cin >> op >> text;
        if ("add" == op) {
            root->insert(text);
        } else if ("find" == op) {
            cout << root->find_partial(text) << endl;
        }
    }
}

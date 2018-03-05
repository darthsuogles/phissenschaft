#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <algorithm>

using namespace std;

class TrieNode: public enable_shared_from_this<TrieNode> {
    vector<shared_ptr<TrieNode>> children_;
    bool is_terminal_;

public:
    TrieNode(bool is_terminal = false)
        : children_(26), is_terminal_(is_terminal) {}

    bool insert(const string &s, int idx = 0) {
        if (is_terminal_) {
            return false;
        }
        if (s.empty() || s.size() == idx) {
            is_terminal_ = true;
            // Check if this string is a proper prefix of any other string
            for (auto node: children_) {
                if (nullptr != node)
                    return false;
            }
            // This string is not a prefix of any other string
            return true;
        }
        char ch = s[idx];
        auto ch_idx = static_cast<int>(ch - 'a');
        auto node = children_[ch_idx];
        if (nullptr == node) {
            node = children_[ch_idx] = make_shared<TrieNode>();
        }
        return node->insert(s, idx + 1);
    }
};

int main() {
    int N; cin >> N;
    vector<string> corpus(N);
    for (int i = 0; i < N; ++i) {
        cin >> corpus[i];
    }

    auto root = make_shared<TrieNode>();
    bool found_prefix_overlap = false;
    for (auto text: corpus) {
        if (!root->insert(text)) {
            found_prefix_overlap = true;
            cout << "BAD SET" << endl;
            cout << text << endl;
            break;
        }
    }
    if (!found_prefix_overlap) {
        cout << "GOOD SET" << endl;
    }
}

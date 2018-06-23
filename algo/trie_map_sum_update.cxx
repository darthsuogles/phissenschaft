#include <vector>
#include <string>
#include <cassert>
#include <iostream>

using namespace std;

class TrieNode {
    vector<TrieNode *> children_;
    bool is_terminal_;
    int val_;
    int sum_;
public:
    TrieNode(): children_(26), is_terminal_(false), sum_(0), val_(0) {}
    int insert_and_get_sum_delta(string key, int val, int idx = 0) {
        int delta = 0;
        if (key.empty() || key.size() == idx) {
            delta = val - (is_terminal_ ? val_ : 0);
            is_terminal_ = true;
            val_ = val;
        } else {
            int pos = key[idx] - 'a';
            assert(0 <= pos && pos < 26);
            TrieNode *next_node = children_[pos];
            if (nullptr == next_node) {
                children_[pos] = next_node = new TrieNode();
            }
            delta = next_node->insert_and_get_sum_delta(key, val, idx + 1);
        }
        sum_ += delta;
        return delta;
    }
    TrieNode *next(char ch) {
        int pos = ch - 'a';
        assert(0 <= pos && pos < 26);
        return children_[pos];
    }
    int sum() { return sum_; }
};

class MapSum {
    TrieNode *root;

public:
    /** Initialize your data structure here. */
    MapSum() {
        root = new TrieNode();
    }

    void insert(string key, int val) {
        root->insert_and_get_sum_delta(key, val);
    }

    int sum(string prefix) {
        auto node = root;
        for (char ch: prefix) {
            if (nullptr == (node = node->next(ch))) return 0;
        }
        return node->sum();
    }
};

/**
 * Your MapSum object will be instantiated and called as such:
 * MapSum obj = new MapSum();
 * obj.insert(key,val);
 * int param_2 = obj.sum(prefix);
 */
int main() {
    MapSum map_sum;
    map_sum.insert("apple", 3);
    map_sum.insert("app", 2);
    cout << map_sum.sum("ap") << endl;
}

#include <iostream>
#include <vector>
#include <stack>
#include <string>
#include <memory>
#include <utility>
#include <queue>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <set>

using namespace std;

set<pair<char, char>> ordering;

class TrieNode {
    bool is_terminal_;
    vector<shared_ptr<TrieNode>> children_;
public:
    TrieNode(): is_terminal_(false), children_(26, nullptr) {}

    void insert(string s, size_t pos = 0) {
        if (s.empty() || pos == s.size()) {
            is_terminal_ = true; return;
        }
        char ch = s[pos];
        int idx = ch - 'a';
        auto node = children_[idx];
        if (nullptr == node) {
            for (int i = 0; i < 26; ++i) {
                if (nullptr == children_[i]) continue;
                ordering.insert(make_pair(i + 'a', ch));
            }
            node = children_[idx] = make_shared<TrieNode>();
        }
        node->insert(s, pos + 1);
    }
};

class Graph {
    vector<vector<bool>> neighbors;
    vector<int> in_degree;
    vector<bool> exists;

public:
    Graph(): neighbors(26, vector<bool>(26, false)),
             in_degree(26, 0),
             exists(26, false) {}

    void add_edge(char ch1, char ch2) {
        int u = ch1 - 'a', v = ch2 - 'a';
        exists[u] = exists[v] = true;
        neighbors[u][v] = true;
        ++in_degree[v];
    }

    // In fact we need a linear order
    void uniq_alphabetical_ordering() {
        // There must be exactly one source and one sink
        int src_cnts= 0, sink_cnts = 0;
        char src_node = '$';
        for (int u = 0; u < 26; ++u) {
            if (!exists[u]) continue;
            if (0 == in_degree[u]) {
                ++src_cnts;
                src_node = u;
            }
            int neighbor_cnts = 0;
            for (int v = 0; v < 26; ++v) {
                if (!exists[v]) continue;
                if (neighbors[u][v]) ++neighbor_cnts;
            }
            if (0 == neighbor_cnts)
                ++sink_cnts;
        }
        if (1 != src_cnts || 1 != sink_cnts) {
            cout << "INVALID" << endl;
            return;
        }

        vector<char> res;
        queue<int> candidates;
        candidates.push(src_node);
        while (!candidates.empty()) {
            int u = candidates.front(); candidates.pop();
            res.push_back(u + 'a');
            for (int v = 0; v < 26; ++v) {
                if (!neighbors[u][v]) continue;
                if (0 == --in_degree[v])
                    candidates.push(v);
            }
        }
        for (auto ch: res)
            cout << ch << " ";
        cout << endl;
    }
};

using graph_t = unordered_map<char, unordered_set<char>>;

/**
 * Compute strongly connected component with augmented DFS
 */
class SCC {
    //
    graph_t &g;
    stack<string> results;
    //
    unordered_map<char, int> DFI;
    int next_dfi;
    unordered_map<char, int> Q;
    unordered_map<char, bool> stacked;
    stack<char> elements;

    inline bool visited(char u) { return -1 != DFI[u]; }

public:
    SCC(graph_t &g): g(g), next_dfi(0) {
        for (auto u_adj: g) {
            char u = get<0>(u_adj);
            stacked[u] = false;
            DFI[u] = -1;
        }
    }

    void dfs(char u) {
        int dfi, q;
        DFI[u] = q = dfi = next_dfi++;
        elements.push(u);
        stacked[u] = true;
        for (auto v: g[u]) {
            if (!visited(v)) {
                dfs(v);
                q = min(q, Q[v]);
            } else if (DFI[v] < dfi && stacked[v]) {
                q = min(q, DFI[v]);
            }
        }
        Q[u] = q;
        if (q == dfi) {
            string res;
            while (!elements.empty()) {
                char v = elements.top();
                elements.pop();
                stacked[v] = false;
                res.push_back(v);
                if (v == u) break;
            }
            cout << "\tSCC block: " << res << endl;
            results.push(res);
        }
    }

    string topo_sort() {
        if (results.empty()) {
            for (auto u_adj: g) {
                char u = get<0>(u_adj);
                if (visited(u)) continue;
                auto adj = get<1>(u_adj);
                dfs(u);
            }
        }
        string res;
        while (!results.empty()) {
            res += results.top();
            results.pop();
        }
        return res;
    }
};

string alienOrder(vector<string>& words) {
	if (words.size() == 1) return words.front();

	unordered_map<char, unordered_set<char>> g;
	for (int i = 1; i < words.size(); i++) {
		string t1 = words[i - 1];
		string t2 = words[i];
		bool found = false;
		for (int j = 0; j < max(t1.length(), t2.length()); j++) {
			if (j < t1.length() && g.count(t1[j]) == 0)
				g.insert(make_pair(t1[j], unordered_set<char>()));
			if (j < t2.length() && g.count(t2[j]) == 0)
				g.insert(make_pair(t2[j], unordered_set<char>()));
			if (j < t1.length() && j < t2.length() && t1[j] != t2[j] && !found) {
				g[t1[j]].insert(t2[j]);
				found = true;
			}
		}
	}

    auto scc = SCC(g);
	return scc.topo_sort();
}


int main() {
    int test_cases; cin >> test_cases;
    for (; test_cases > 0; --test_cases) {
        int n; cin >> n;
        // ordering.clear();
        // auto root = make_shared<TrieNode>();

        vector<string> words;
        for (int i = 0; i < n; ++i) {
            string word; cin >> word;
            //root->insert(word);
            words.push_back(word);
        }
        cout << alienOrder(words) << endl;

        // Graph g;
        // for (auto kv: ordering)
        //     g.add_edge(get<0>(kv), get<1>(kv));
        // g.uniq_alphabetical_ordering();
    }
}

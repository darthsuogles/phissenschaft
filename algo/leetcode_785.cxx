#include <iostream>
#include <vector>
#include <unordered_set>

using namespace std;

class Solution {
public:
    bool isBipartite(vector<vector<int>> &adj_list) {
		if (adj_list.empty()) return true;
		vector<int> color(adj_list.size(), -1);
		for (int root = 0; root < adj_list.size(); ++root) {
			if (-1 != color[root]) continue;
			unordered_set<int> queue;
			queue.insert(root);
			color[root] = 0;
			while (!queue.empty()) {
				auto it = queue.begin(); int u = *it; queue.erase(it);
				int u_color = color[u];
				for (int v: adj_list[u]) {
					// Conflict detected
					if (u_color == color[v]) return false;
					// Visited already
					if (-1 != color[v]) continue;
					color[v] = 1 - u_color;
					queue.insert(v);
				}
			}
		}
		return true;
    }
};

Solution sol;

void TEST(vector<vector<int>> adj_list, bool tgt) {
	auto res = sol.isBipartite(adj_list);
	if (res != tgt) {
		cerr << "FAIL: " << res << " != (ref) " << tgt << endl;
	} else {
		cout << "PASS" << endl;
	}
}

int main() {
	TEST({{},{2,4,6},{1,4,8,9},{7,8},{1,2,8,9},{6,9},{1,5,7,8,9},{3,6,9},{2,3,4,6,9},{2,4,5,6,7,8}}, false);
	TEST({{1, 3}, {0, 2}, {1, 3}, {0, 2}}, true);
	TEST({{1,2,3}, {0,2}, {0,1,3}, {0,2}}, false);
}

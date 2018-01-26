/**
 * Friend circles
 */

#include <iostream>
#include <vector>
#include <unordered_set>

using namespace std;

class UnionFind {
	int n;
	vector<int> _rank;
	vector<int> _parent;

public:
	UnionFind(int n): n(n) {
		_rank.reserve(n);
		_parent.reserve(n);
		for (int u = 0; u < n; ++u) {
			_rank.push_back(1);
			_parent.push_back(u);
		}
	}

	int find(int u) {
		int r = _parent[u];
		while (r != _parent[r])
			r = _parent[r];
		while (u != r) {
			int tmp = _parent[u];
			_parent[u] = r;
			u = tmp;
		}
		return r;
	}

	void combine(int u, int v) {
		int pu = find(u), pv = find(v);
		if (pu == pv) return;
		int ru = _rank[pu], rv = _rank[pv];
		if (ru == rv) {
			_parent[pu] = pv;
			_rank[pv] += 1;
		} else if (rv < ru)
			_parent[pv] = pu;
		else
			_parent[pu] = pv;
	}

	int num_components() {
		unordered_set<int> seen;
		for (auto u: _parent) {
			seen.insert(find(u));
		}
		return seen.size();
	}
};

int findCircleNum(vector< vector<int> > &M) {
	if (M.empty()) return 0;
	int n = M.size();  // n-by-n matrix	
	UnionFind uf(n);
	for (int u = 0; u < n; ++u)
		for (int v = u + 1; v < n; ++v) 
			if ( 1 == M[u][v] )
				uf.combine(u, v);
	return uf.num_components();
}

void TEST(vector< vector<int> > M, int expected) {
	int res = findCircleNum(M);
	if (res != expected) {
		cout << "ERROR " << res << " but expect " << expected << endl;
	} else
		cout << "Ok" << endl;
}

int main() {
	TEST({{1, 1, 0}, {1, 1, 1},{0, 1, 1}}, 1);
	TEST({{1, 1, 0}, {1, 1, 0},{0, 0, 1}}, 2);
	TEST({{1, 0, 0, 1},
		  {0, 1, 1, 0},
		  {0, 1, 1, 1},
		  {1, 0, 1, 1}}, 1);
}

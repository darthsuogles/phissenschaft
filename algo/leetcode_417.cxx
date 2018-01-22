#include <iostream>
#include <vector>
#include <queue>
#include <cassert>
#include <utility>

using namespace std;

vector< pair<int, int> > pacificAtlantic(vector< vector<int> >& matrix) {
	vector< pair<int, int> > res;
	int m = matrix.size(); if (0 == m) return res;
	int n = matrix[0].size(); if (0 == n) return res;

	vector< vector<bool> > tbl_pac(m, vector<bool>(n, false));

	queue< pair<int, int> > search_cands;
	for (int j = 0; j < n; ++j) {
		search_cands.push(make_pair(0, j));
	}
	for (int i = 1; i < m; ++i) {
		search_cands.push(make_pair(i, 0));
	}

	while (! search_cands.empty()) {
		auto pr = search_cands.front(); search_cands.pop();
		int i = pr.first, j = pr.second;		
		if (tbl_pac[i][j]) continue;
		tbl_pac[i][j] = true;
		int curr = matrix[i][j];
		for (int ii = max(0, i-1); ii <= min(m-1, i+1); ++ii) {
			for (int jj = max(0, j-1); jj <= min(n-1, j+1); ++jj) {
				if (ii != i && jj != j) continue;
				if (tbl_pac[ii][jj]) continue;
				if (matrix[ii][jj] >= curr) 
					search_cands.push(make_pair(ii, jj));
			}
		}
	}

	// Atlantic to Pacific
	vector< vector<bool> > tbl_atl(m, vector<bool>(n, false));
	for (int j = 0; j < n; ++j) {
		search_cands.push(make_pair(m-1, j));
	}
	for (int i = 0; i + 1 < m; ++i) {
		search_cands.push(make_pair(i, n-1));
	}
	while (! search_cands.empty()) {
		auto pr = search_cands.front(); search_cands.pop();
		int i = pr.first, j = pr.second;
		if (tbl_atl[i][j]) continue;
		tbl_atl[i][j] = true;
		int curr = matrix[i][j];
		for (int ii = max(0, i-1); ii <= min(m-1, i+1); ++ii) {
			for (int jj = max(0, j-1); jj <= min(n-1, j+1); ++jj) {
				if (ii != i && jj != j) continue;
				if (tbl_atl[ii][jj]) continue;
				if (matrix[ii][jj] >= curr) 
					search_cands.push(make_pair(ii, jj));
			}
		}
	}

	// Finalize
	for (int i = 0; i < m; ++i)	
		for (int j = 0; j < n; ++j) 
			if (tbl_atl[i][j] && tbl_pac[i][j])
				res.push_back(make_pair(i, j));
	return res;
}

int main() {
	auto matrix = vector< vector<int> > {
		{1,2,2,3,5}, 
		{3,2,3,4,4},
		{2,4,5,3,1},
		{6,7,1,4,5},
		{5,1,1,2,4}
	};

	auto res = pacificAtlantic(matrix);
	for (auto& pr: res) 
		cout << "[" << pr.first << ", " << pr.second << "]" << endl;
}

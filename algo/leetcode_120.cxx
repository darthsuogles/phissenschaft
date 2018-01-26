#include <iostream>
#include <vector>

using namespace std;

int minimumTotal(vector< vector<int> >& triangle) {
	auto n = triangle.size();
	if (0 == n) return 0;

	vector<int> level_opt(n, INT_MAX);	
	if (triangle[0].empty()) return 0;	
	level_opt[0] = triangle[0][0];

	for (int i = 1; i < n; ++i) {  // each level
		auto& slice = triangle[i];
		int prev = level_opt[0];  // DP initial condition
		for (int j = 0; j <= i; ++j) {
			int curr = level_opt[j];
			level_opt[j] = slice[j] + min(prev, curr);
			prev = curr;
		}
	}

	int res = INT_MAX;
	for (auto a: level_opt) res = min(res, a);
	return res;
}

int main() {
	auto triangle = vector< vector<int> > {
		{2}, {3, 4}, {6, 5, 7}, {4, 1, 8, 3}
	};
	int res = minimumTotal(triangle);
	cout << res << endl;
}

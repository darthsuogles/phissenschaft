#include <iostream>
#include <vector>
#include <cassert>

using namespace std;

int splitArray(vector<int>& nums, int m) {
	auto n = nums.size();
	if (0 == n) return 0;
	vector< vector<int> > tbl(m, vector<int>(n, INT_MAX));

	vector<int> sum_avant(n, 0);
	vector<int> sum_apres(n, 0);
	int avant = 0, apres = 0;
	for (int i = 0, j = n - 1; i < n; ++i, --j) {
		sum_avant[i] = avant; avant += nums[i];
		sum_apres[j] = apres; apres += nums[j];
	}
	int sum_tot = sum_apres[0] + nums[0];

	for (int j = 0; j < n; ++j) 
		tbl[0][j] = sum_avant[j] + nums[j];

	for (int i = 1; i < m; ++i) {  // each split level i+1
		auto& tbl_prev = tbl[i-1];
		auto& tbl_curr = tbl[i];
		for (int j = i; j < n; ++j) {
			int curr_opt = INT_MAX;
			int partial_base = sum_tot - sum_apres[j];
			for (int k = 0; k < j; ++k) {
				int partial = partial_base - sum_avant[k+1];
				int sub_val = max(tbl_prev[k], partial);
				//cout << i+1 << ": 1 ... " << j+1 << " sub: " << sub_val << endl;
				curr_opt = min(curr_opt, sub_val);
			}
			tbl_curr[j] = curr_opt;
		}
	}

	return tbl[m-1][n-1];
}

int main() {
	auto nums = vector<int> {7, 2, 5, 10, 8};
	int m = 2;
	int res = splitArray(nums, m);
	cout << res << endl;
}

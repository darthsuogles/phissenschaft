#include <iostream>
#include <cassert>
#include <fstream>
#include <vector>

using namespace std;

class Solution {
	using integer = long long;
public:
    int numSubarrayProductLessThanK(vector<int>& nums, int k) {
		if (nums.empty() || k <= 1) return 0;
		nums.push_back(k);
		const int n = nums.size();
		integer prod_max = static_cast<integer>(k);
		integer prod = 1;
		integer num_subs = 0;

		// Far easier mantaining the right end than with the left end.
		for (int i = 0, j = 0; j < n; ++j) {
			prod *= nums[j];
			while (prod >= prod_max) prod /= nums[i++];
			num_subs += j - i + 1;
		}
		return static_cast<int>(num_subs);
    }
};

Solution sol;

void TEST(vector<int> nums, int k, int cnts) {
	auto res = sol.numSubarrayProductLessThanK(nums, k);
	if (cnts == res) {
		cout << "PASS" << endl;
	} else {
		cout << "FAIL: " << res << " != (ref) " << cnts << endl;
	}
}

int main() {
	TEST({10, 5, 2, 6}, 100, 8);
	TEST({10, 5, 2, 6, 10}, 100, 10);
	TEST({100}, 100, 0);
	TEST({1, 1, 1}, 2, 6);
	TEST({1, 1, 1, 1, 1}, 5, 15);
	TEST({1, 1, 5, 6, 1}, 5, 4);

	// fstream fin("IN713");
	// vector<int> vals;
	// int a; while (fin >> a) vals.push_back(a);
	// TEST(vals, 5, 367968907);
}

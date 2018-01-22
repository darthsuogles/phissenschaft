#include <iostream>
#include <vector>
#include <stack>
#include <tuple>
#include <unordered_map>

using namespace std;

bool search_config(vector<int>& nums, int i, vector<int>& sides) {
	bool all_sides_filled = true;
	for (auto a: sides) {
		if (0 != a) { all_sides_filled = false; break; }
	}
	if (nums.size() == i) return all_sides_filled;
	if (all_sides_filled) return false;

	int curr = nums[i];
	int k; 
	for (int j = 0; j < 4; ++j) {
		if (sides[j] < curr) continue;
		for (k = 0; k < j; ++k) 
			if (sides[k] == sides[j]) break;
		if (k < j) continue;
		sides[j] -= curr;
		if (search_config(nums, i + 1, sides)) 
			return true;
		sides[j] += curr;
	}
	return false;
}

bool makesquare(vector<int>& nums) {
	if (nums.empty()) return false;
	long tot = 0;
	for (auto a: nums) tot += a;
	if (0 != (tot % 4)) return false;
	int side_len = tot / 4;
	
	sort(nums.begin(), nums.end(), greater<int>());
	vector<int> sides(4, side_len);
	return search_config(nums, 0, sides);
}


void TEST(vector<int> nums, bool tgt) {
	if (tgt == makesquare(nums))
		cout << "OK" << endl;
	else 
		cout << "ERR" << endl;
}

int main() {
	
	TEST({1,1,2,2,2}, true);
	TEST({3,3,3,3,4}, false);
	TEST({5,5,5,5,4,4,4,4,3,3,3,3}, true);
}

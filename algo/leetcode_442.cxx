#include <iostream>
#include <vector>

using namespace std;

vector<int> findDuplicates(vector<int>& nums) {
	vector<int> res;
	for (int i = 0; i < nums.size(); ++i) {
		int a = nums[i];
		if (i + 1 == a) continue;
		nums[i] = -1;
		while (true) {
			int tmp = nums[a - 1];
			if (tmp == a) {
				res.push_back(a);
				break;
			}
			nums[a - 1] = a;
			if (tmp <= 0) break;
			a = tmp;
		}
	}
	return res;
}

void TEST(vector<int> nums, vector<int> tgt) {
	auto res = findDuplicates(nums);
	sort(res.begin(), res.end());
	sort(tgt.begin(), tgt.end());
	cout << "RESULT: ";
	for (auto a: res) cout << a << " ";
	cout << endl;
	if (res.size() != tgt.size()) {
		cout << "ERROR" << endl;
		return;
	}
	int i;
	for (i = 0; i < tgt.size(); ++i) {
		if (tgt[i] != res[i]) {
			cout << "ERROR"; break;
		}
	}	
	if (i == tgt.size())
		cout << "OK";
	cout << endl;
}

int main() {
	TEST({4,3,2,7,8,2,3,1}, {2,3});
	TEST({10,2,5,10,9,1,1,4,3,7}, {10, 1});
}

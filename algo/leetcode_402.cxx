#include <iostream>
#include <unordered_set>
#include <vector>
#include <utility>
#include <string>

using namespace std;

class Solution {
public:
	string removeKdigits(string num_repr, int k) {
		while (--k >= 0 && !num_repr.empty()) {
			const int n = num_repr.size();
			int i = 0;
			for (; i + 1 < n; ++i) {
				if (num_repr[i] > num_repr[i + 1]) break;
			}
			num_repr =
				num_repr.substr(0, i) +
				num_repr.substr(i + 1, n - i - 1);
		}
		int i = 0;
		for (; i < num_repr.size() && '0' == num_repr[i]; ++i);
		if (num_repr.size() == i) return "0";
		return num_repr.substr(i, num_repr.size() - i);
	}

    string removeKdigitsBFS(string num_repr, int k) {
		if (num_repr.empty() || 0 == k) return num_repr;
		unordered_set<string> curr_queue, next_queue;
		curr_queue.insert(num_repr);
		while (--k >= 0) {
			while (!curr_queue.empty()) {
				auto itop = curr_queue.begin();
				string curr = *itop;
				curr_queue.erase(itop);
				for (int i = 0; i < curr.size(); ++i) {
					next_queue.insert(curr.substr(0, i) +
									  curr.substr(i + 1, curr.size() - i - 1));
				}
			}
			swap(curr_queue, next_queue);
		}
		int min_val = 1e9;
		string min_repr;
		for (auto repr: curr_queue) {
			int val = 0;
			for (auto ch: repr) {
				val = val * 10 + static_cast<int>(ch - '0');
			}
			if (val < min_val) {
				min_val = val;
				min_repr = repr;
			}
		}
		int i = 0;
		for (; i < min_repr.size() && '0' == min_repr[i]; ++i);
		if (min_repr.size() == i) return "0";
		return min_repr.substr(i, min_repr.size() - i);
    }
};

Solution sol;

void TEST(string repr, int k, string tgt) {
	auto res = sol.removeKdigits(repr, k);
	if (res != tgt) {
		cout << "FAIL: " << res << " != (ref) " << tgt << endl;
	} else {
		cout << "PASS" << endl;
	}
}

int main() {
	TEST("1432219", 3, "1219");
	TEST("10200", 1, "200");
	TEST("10", 2, "0");
	TEST("100", 1, "0");
	TEST("112", 1, "11");
}

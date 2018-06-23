#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

class Solution {
public:
    int leastInterval(vector<char>& tasks, int n) {
		int num_tasks = static_cast<int>(tasks.size());
		if (0 == n) return num_tasks;
		int cntr[26] = {0};
		for (int i = 0; i < 26; cntr[i++] = 0);
		for (auto task: tasks) {
			++cntr[task - 'A'];
		}
		int max_cnts = 0, max_val_reps = 0;
		for (int i = 0; i < 26; ++i) {
			int cnts = cntr[i];
			if (cnts == max_cnts) {
				++max_val_reps;
			} else if (cnts > max_cnts) {
				max_cnts = cnts;
				max_val_reps = 1;
			}
		}
		int stride_size = ceil(num_tasks / static_cast<double>(n + 1));
		int stride_reps = 1;
		if (max_cnts >= stride_size) {
			stride_size = max_cnts;
			stride_reps = max_val_reps;
		}
		//cout << stride_reps << endl;
		int tot_spaces = (stride_size - 1) * (n + 1) + stride_reps;
		int last_rem = max(0,
						   num_tasks
						   - stride_size * stride_reps
						   - (stride_size - 1) * (n + 1 - stride_reps));
		return tot_spaces + last_rem;
    }
};

Solution sol;

void TEST(vector<char> tasks, int n, const int tgt) {
	int res = sol.leastInterval(tasks, n);
	if (res == tgt) {
		cout << "PASS" << endl;
	} else {
		cout << "FAIL: " << res << " != " << tgt << endl;
	}
}

int main() {
	TEST({'A', 'A', 'A', 'B', 'B', 'B'}, 2, 8);
	TEST({'A', 'A', 'A', 'B', 'B', 'B'}, 0, 6);
	TEST({'A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E'}, 4, 10);
	TEST({'A', 'A', 'A', 'A', 'A', 'A', 'B', 'C', 'D', 'E', 'F', 'G'}, 2, 16);
}

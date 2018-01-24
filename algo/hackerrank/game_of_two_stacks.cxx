#include <vector>
#include <algorithm>
#include <iostream>
#include <cassert>
<<<<<<< HEAD
#include <stack>
=======
>>>>>>> a292468... daily checkup
#include <iterator>

using namespace std;

<<<<<<< HEAD
void linear_search() {
    int g; cin >> g;
    using integer = long long;
    for (int game = 0; game < g; ++game) {
        int n, m; integer x;
        cin >> n >> m >> x;

        bool skip_rest = false;
        int a;
        integer psum = 0L;
        int reward = 0;
        stack<int> stack;
        for (int i = 0; i < n; ++i) {
            cin >> a;
            if (skip_rest) continue;
            if (psum + a > x) { skip_rest = true; continue; }
            stack.push(a);
            ++reward;
            psum += a;
        }
        int max_reward = reward;
        integer tot_sum = psum;
        skip_rest = false;
        for (int j = 0; j < m; ++j) {
            cin >> a;
            if (skip_rest) continue;
            tot_sum += a;
            while (tot_sum > x && !stack.empty()) {
                tot_sum -= stack.top();
                reward -= 1;
                stack.pop();
            }
            if (tot_sum <= x) reward += 1;
            else skip_rest = true;
            max_reward = max(reward, max_reward);
        }
        cout << max_reward << endl;
    }
}

void prefix_array_binary_search() {
=======
int main() {
>>>>>>> a292468... daily checkup
    int g; cin >> g; // number of games
    using integer = long long;
    for (int game = 0; game < g; ++game) {
        int n, m; integer x;
        cin >> n >> m >> x;
        vector<integer> A, B;
        integer psum, a;
        bool over_flag;

        // Be aware of integer overflow
        psum = 0L; over_flag = false;
        for (int i = 0; i < n; ++i) {
            cin >> a;
            if (over_flag) continue;
            psum += a;
            if (psum > x) over_flag = true;
            else A.push_back(psum);
        }

        psum = 0L; over_flag = false;
        for (int j = 0; j < m; ++j) {
            cin >> a;
            if (over_flag) continue;
            psum += a;
            if (psum > x) over_flag = true;
            else B.push_back(psum);
        }

        auto bv = upper_bound(B.begin(), B.end(), x);
        int max_val = distance(B.begin(), bv);
        for (int i = 0; i < A.size(); ++i) {
            // Then we should binary search into B
            integer target = x - A[i];
            assert(target >= 0L);
            auto bv = upper_bound(B.begin(), B.end(), target);
            max_val = max(i + 1 + int(distance(B.begin(), bv)), max_val);
        }
        cout << max_val << endl;
    }
}
<<<<<<< HEAD

int main() {
    linear_search();
}
=======
>>>>>>> a292468... daily checkup

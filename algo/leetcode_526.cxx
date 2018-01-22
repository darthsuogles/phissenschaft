/**
 * Count number of "beautiful arrangements"
 */
#include <iostream>
#include <unordered_map>
#include <vector>
using namespace std;

int search(int a, int N, vector< vector<int> > &pos_avail, vector<bool> &slot_used) {
    if (a > N) {
        for (int k = 0; k < N; ++k)
            if (! slot_used[k]) return false;
        return true;
    }
    
    int cnt = 0;
    for (auto i: pos_avail[a]) {
        if (slot_used[i]) continue;
        slot_used[i] = true;
        cnt += search(a + 1, N, pos_avail, slot_used);
        slot_used[i] = false;
    }
    return cnt;
}

int find_sub(int a, vector<int> &nums) {
    if (a < 1) return 1;
    int cnts = 0;
    for (int k = 0; k < a; ++k) {
        if (0 == nums[k] % a || 0 == a % nums[k]) {
            swap(nums[k], nums[a-1]);
            cnts += find_sub(a - 1, nums);
            swap(nums[k], nums[a-1]);
        }
    }
    return cnts;
}

int countArrangement(int N) {
    vector<int> nums;
    for (int i = 1; i <= N; ++i) nums.push_back(i);
    return find_sub(N, nums);
}

int countArrangementV1(int N) {
    // locate possible place for an integer
    vector< vector<int> > pos_avail(N + 1, vector<int>());  
    for (int a = 1; a <= N; ++a) {
        for (int i = 1; i <= N; ++i) {
            if (0 == a % i || 0 == i % a)
                pos_avail[a].push_back(i);
        }
    }
    
    vector<bool> slot_used(N + 1, false);
    slot_used[0] = true;
    return search(1, N, pos_avail, slot_used);
}

int main() {
    for (int i = 1; i <= 15; ++i)
        cout << i << " => " << countArrangement(i) << endl;
}

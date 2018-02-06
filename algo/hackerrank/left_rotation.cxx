#include <iostream>
#include <vector>

using namespace std;

int main() {
    int N, k;
    cin >> N >> k;
    k = k % N;
    vector<int> nums(N);
    for (int i = 0; i < N; ++i) cin >> nums[i];
    reverse(nums.begin(), nums.end());
    reverse(nums.begin(), nums.begin() + N - k);
    reverse(nums.begin() + N - k, nums.end());
    for (auto a: nums) cout << a << " ";
    cout << endl;
}

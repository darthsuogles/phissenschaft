#include <vector>
#include <iostream>
#include <utility>

using namespace std;

template <typename T>
pair<int, int> three_way_partition(vector<T> &nums) {
    auto n = nums.size();
    if (n < 1) return make_pair(0, 0);
    auto pivot = nums[0];
    int i = 0, j = 0, k = n - 1;
    while (j <= k) {
        if (nums[j] < pivot) {
            swap(nums[i++], nums[j++]);
        } else if (nums[j] > pivot) {
            swap(nums[j], nums[k--]);
        } else {
            ++j; // if j == k and nums[j] == pivot, incr j
        }
    }
    // After the while loop
    // - i points to the first pivot
    // - j points to the first after pivot region
    // - j could point to one element after the bound
    return make_pair(i, j);
}

int main() {
    vector<int> nums(100);
    for (int i = 0; i < nums.size(); ++i) {
        nums[i] = (i + int(1e7)) % 37;
    }
    auto ij = three_way_partition(nums);
    auto i = get<0>(ij), j = get<1>(ij);

    int k = 0;
    for (; k < i; ++k) cout << nums[k] << " ";
    cout << endl;
    for (; k < j; ++k) cout << nums[k] << " ";
    cout << endl;
    for (; k < nums.size(); ++k) cout << nums[k] << " ";
    cout << endl;
    cout << "i " << i << " j " << j << endl;
    return 0;
}

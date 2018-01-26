#include <iostream>
#include <vector>

using namespace std;

/**
 * Assuming we are given a list of distinct integers
 */
int first_missing_nat(int *nums, int n, int lo = 0) {
    if (NULL == nums || 0 == n) return lo + 1;
    
    // Observe that the first missing number <= n + 1
    int i = 0, j = n - 1;
    int mid = (i + j) / 2;
    int pivot = nums[mid]; nums[mid] = nums[0]; nums[0] = pivot;
    while (true) {
        while (i < n && nums[i] <= pivot) ++i;
        while (nums[j] > pivot) --j;
        if (j <= i) break;
        int tmp = nums[j]; nums[j] = nums[i]; nums[i] = tmp;
    }
    //for (int k = 0; k <= j; ++k) cout << nums[k] << " "; cout << endl;
    int m = j + 1; // # elements <= pivot
    if (lo + m == pivot)
        return first_missing_nat(nums + m, n - m, pivot);
    else // pivot > m, missing number in first half
        return first_missing_nat(nums + 1, m - 1, lo);
}

int main() {
    auto arr = vector<int> {1, 4, 5, 6};
    int res = first_missing_nat(&arr[0], arr.size());
    cout << res << endl;
}

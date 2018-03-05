#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

int kth_element(int *arr, int n, int k) {
    if (k > n) throw runtime_error("k too large");
    int i = 0, j = 0, t = n - 1;
    int pivot = arr[0];
    while (j <= t) {
        if (arr[j] < pivot) {
            swap(arr[i++], arr[j++]);
        } else if (arr[j] > pivot) {
            swap(arr[j], arr[t--]);
        } else {
            ++j;
        }
    }
    // The middle "j - i" elements are equal to the pivot
    if (i < k && k <= j) return pivot;
    // The first "i" elements smaller than the pivot
    if (k <= i) return kth_element(arr, i, k);
    // The last "n - j" elements are larger than the pivot
    if (k > j) return kth_element(arr + j, n - j, k - j);
}

int get_kth_smallest(int *arr, int n, int k) {
    if (n < k) throw runtime_error("k too large");

    // Fixing the range
    int i = 0, j = n - 1;
    // Selecting a pivot element randomly within
    default_random_engine prng;
    uniform_int_distribution<int> distr(i, j);
    // Exchange pivot element with the first in array
    swap(arr[0], arr[distr(prng)]);
    int pivot = arr[0];

    // The prefix until j-th element are <= pivot
    // a0 (= pivot) a1 a2 .... aj || ...
    while (true) {
        while (i < n && arr[i] <= pivot) ++i; // could be i == n
        while (arr[j] > pivot) --j;
        if (j <= i) break;  // arr[j] <= pivot
        swap(arr[i], arr[j]);
    }
    int m = j + 1;
    if (m == k)  // m is the number of elements <= pivot
        return pivot;
    else if (m < k)
        return get_kth_smallest(arr + m, n - m, k - m);
    else
        return get_kth_smallest(arr + 1, j, k);
}

int main() {
    const int n = 100;
    default_random_engine prng;
    uniform_int_distribution<int> distr(-100, 100);
    vector<int> arr;
    for (int i = 0; i < n; ++i)
        arr.push_back(distr(prng));

    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    try {
        int n = arr.size();
        for (int k = 1; k <= n; ++k) {
            shuffle(arr.begin(), arr.end(), default_random_engine(seed));
            //int elem = get_kth_smallest(&arr[0], n, k);
            int elem = kth_element(&arr[0], n, k);
            sort(arr.begin(), arr.end());
            int ref = arr[k - 1];
            if (ref != elem)
                cerr << "ERROR " << elem << " but expect " << ref << endl;
            else
                cout << "Ok" << endl;
        }
    } catch (...) {
        cerr << "Error" << endl;
    }
    return 0;
}

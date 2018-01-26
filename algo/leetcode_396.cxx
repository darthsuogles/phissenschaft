#include <iostream>
#include <vector>

using namespace std;

int maxRotateFunction(vector<int> &A) {
    int n = A.size();
    int res = 0;
    int arr_sum = 0;
    for (int i = 0; i < n; ++i) {        
        int a = A[i];
        arr_sum += a;
        res += i * a;
    }
    int curr = res;
    for (int i = n-1; i >= 0; --i) {
        curr += arr_sum - A[i] * n;
        res = max(res, curr);
    }
    return res;
}

#define TEST(...) { \
    vector<int> arr = {__VA_ARGS__}; \
    cout << maxRotateFunction(arr) << endl; }

int main() {
    TEST(4, 3, 2, 6);
    TEST(4, 15, 14, 3, 14,-8,12,-9,17,-1,15,1,10,19,-7,15,8,7,-8,11);
}

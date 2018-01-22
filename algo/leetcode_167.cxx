#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;

int binary_search(vector<int>& arr, int tgt) {
    if (arr.empty()) return -1;
    if (tgt < arr[0]) return -1;
    int i = 0, j = arr.size() - 1;

    while (i + 1 < j) {
        int k = (i + j) / 2;
        int v = arr[k];
        if (v == tgt) return k;  // depends
        if (v < tgt)
            i = k; 
        else 
            j = k;
    }
    if (arr[j] <= tgt) return j;
    if (arr[i] <= tgt) return i;
    return -1;
}


vector<int> twoSum(vector<int>& numbers, int target) {
    vector<int> res;
    if (numbers.empty()) return res;
    int idx_last = binary_search(numbers, target - numbers[0]);
    unordered_map<int, int> tbl;
    for (int i = 0; i <= idx_last; ++i) {
        int a = numbers[i];
        if (tbl.count(a) > 0) {
            res.push_back(1 + tbl[a]);
            res.push_back(1 + i);
            break;
        }
        tbl[target - a] = i;
    }
    return res;
}

void TEST(vector<int> arr, int target) {
    for (auto idx: twoSum(arr, target)) cout << idx << " ";
    cout << endl;    
}

int main() {
    TEST({2, 7, 11, 15}, 9);
    TEST({-3, 3, 4, 90}, 0);
}

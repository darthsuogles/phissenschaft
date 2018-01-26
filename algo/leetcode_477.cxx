#include <iostream>
#include <vector>

using namespace std;

class Solution {
private: 
    static const int bitlen = 256;
    int hamm_tbl[bitlen];
    static const int mask = bitlen - 1;
    static const int nbits = 8;

    int hamm_dist(int a, int b) {
        int res = 0;
        for (int shft = 0; shft < 4; ++shft, a >>= nbits, b >>= nbits)
            res += hamm_tbl[(a & mask) ^ (b & mask)];
        return res;
    }

public:
    Solution() {
        for (int d = 0; d < bitlen; ++d) {
            int cnt = 0;
            for (int val = d; val > 0; val >>= 1) 
                cnt += val & 1;
            hamm_tbl[d] = cnt;
        }       
    }

public:
    int totalHammingDistanceBrute(vector <int>& nums) {
        // Build a table of 256 
        int res = 0;
        for (int i = 0; i + 1 < nums.size(); ++i) 
            for (int j = i + 1; j < nums.size(); ++j) 
                res += hamm_dist(nums[i], nums[j]);
        return res;
    }

    // Just find number of positive bits occurrence for each
    int totalHammingDistance(vector<int>& nums) {
        int bit_cntr[32] = {0};
        for (auto a : nums) {            
            for (int i = 0; a > 0; a >>= 1, ++i)
                bit_cntr[i] += a & 1;
        }
        int res = 0;
        for (int i = 0; i < 32; ++i) {
            int curr = bit_cntr[i];
            res += curr * (nums.size() - curr);
        }
        return res;
    }
};

void TEST(vector<int> nums, int tgt) {   
    Solution sol;
    int res = sol.totalHammingDistance(nums);
    if (res != tgt)
        cout << "ERR: " << res << " != " << tgt << endl;
    else
        cout << "OK" << endl;
}

int main() {
    
    TEST({4, 14, 2}, 6);
}

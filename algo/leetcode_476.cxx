#include <iostream>

using namespace std;

int findComplement0(int num) {
    int tmp = 0, res = 0, cnt = 0;
    for (; num > 0; num >>= 1, ++cnt) {
        tmp = (tmp << 1) ^ (!(num & 1));
    }
    for (; cnt-- > 0; tmp >>= 1) {
        res = (res << 1) ^ (tmp & 1); 
    }
    return res;
}

// Using two's complement
int findComplement(int num) {
    int tmp = num, mask = 1;
    for (; tmp > 0; tmp >>= 1, mask <<= 1);
    --mask;
    return (-num - 1) & mask;
}

int main() {
    cout << findComplement(5) << endl;
    cout << findComplement(1) << endl;
}

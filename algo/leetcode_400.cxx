#include <iostream>

using namespace std;

int findNthDigit(int n) {
    int d;
    long m, base;
    long cmp = n;
    long csum = 0;
    for (d = 1, m = 9, base = 1; 
         ;
         ++d, m *= 10, base *= 10) {
        csum += d * m;
        if (csum == cmp) return 9;
        if (csum > cmp) {
            csum -= d * m;
            break;
        }
    }
    int kd = d - (cmp - 1 - csum) % d;  // which digit
    long num = base + (cmp - 1 - csum) / d; // which number
    //cout << num << " from " << base << " k-th " << kd << endl;
    for (; --kd > 0; num /= 10);
    return num % 10;
}

int main() {
    cout << findNthDigit(3) << endl;
    cout << findNthDigit(9) << endl;
    cout << findNthDigit(11) << endl;
    cout << findNthDigit(1000000000) << endl;
    cout << findNthDigit(2147483647) << endl;
}

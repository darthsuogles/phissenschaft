#include <iostream>

using namespace std;

bool isBadVersion(int version) {
    //return (version >= 2147483647);
    return (version >= 2);
}

int bvs(int m, int n) {
    if (m >= (n - 1)) 
        return isBadVersion(m) ? m : n;
    int k = m + (n - m) / 2;
    return isBadVersion(k) ? bvs(m, k) : bvs(k+1, n);
}

int firstBadVersion(int n) {
    return bvs(1, n);
}

int main() {
    //cout << firstBadVersion(2147483647) << endl;
    cout << firstBadVersion(2) << endl;   
}

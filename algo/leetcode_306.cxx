#include <iostream>

using namespace std;

bool check_seq(long a, long b, string num, int init) {
    if (num.empty() || init == num.size()) return false;
    if ('0' == num[init]) return false;

    long curr = 0;  
    bool is_match = false;
    for (int i = init; i < num.size(); ++i) {
        int d = num[i] - '0';        
        if (is_match) {
            if (0 == d) return false;
            a = b; b = curr; curr = 0; init = i;
        }
        curr = curr * 10 + d;
        if (a + b < curr) return false;
        is_match = (a + b) == curr;
    }
    return is_match;
}

bool isAdditiveNumber(string num) {
    long a = 0;
    for (int i = 0; i + 2 < num.size(); ++i) {
        if (0 < i && 0 == a) return false;
        a = a * 10 + (num[i] - '0');
        long b = 0;
        for (int j = i + 1; j + 1 < num.size(); ++j) {
            if (i + 1 < j && 0 == b) continue;
            b = b * 10 + (num[j] - '0');
            if (check_seq(a, b, num, j + 1))
                return true;
        }
    }
    return false;
}


int main() {
    cout << isAdditiveNumber("112358") << endl;
    cout << isAdditiveNumber("199100199") << endl;
    cout << isAdditiveNumber("12") << endl;
    cout << isAdditiveNumber("111") << endl;
    cout << isAdditiveNumber("101") << endl;
    cout << isAdditiveNumber("1023") << endl;
    cout << isAdditiveNumber("0235813") << endl;
    cout << isAdditiveNumber("198019823962") << endl;
    cout << isAdditiveNumber("121474836472147483648") << endl;
}

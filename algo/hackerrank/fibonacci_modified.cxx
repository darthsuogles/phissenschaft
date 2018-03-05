#include <iostream>
#include <memory>
#include <vector>

using namespace std;

class HighPrecInt: public enable_shared_from_this<HighPrecInt> {
    vector<unsigned int> digits; // left to right
public:
    HighPrecInt() {}

    HighPrecInt(int num) {
        while (num > 0) {
            digits.push_back(num % 10);
            num /= 10;
        }
    }

    shared_ptr<HighPrecInt> add(shared_ptr<HighPrecInt> other) {
        auto n = digits.size();
        auto m = other->digits.size();
        if (n < m) return other->add(shared_from_this());
        auto &ds0 = digits;
        auto &ds1 = other->digits;

        auto res = make_shared<HighPrecInt>();
        unsigned int carry = 0;
        int i = 0;
        auto &res_digits = res->digits;
        for (; i < n; ++i) {
            auto d = carry + ds0[i] + (i < m ? ds1[i] : 0);
            carry = d >= 10;
            res_digits.push_back(d % 10);
        }
        if (carry > 0)
            res_digits.push_back(carry);
        return res;
    }

    shared_ptr<HighPrecInt> pow2() {
        auto res = make_shared<HighPrecInt>();
        auto &res_digits = res->digits;
        unsigned int carry = 0;
        int shift = 0;
        for (auto a: digits) {
            int idx = shift;
            for (auto b: digits) {
                unsigned int val;
                if (idx < res_digits.size()) {
                    val = a * b + carry + res_digits[idx];
                    res_digits[idx] = val % 10;
                } else {
                    val = a * b + carry;
                    res_digits.push_back(val % 10);
                }
                carry = val / 10;
                ++idx;
            }
            if (carry != 0)
                res_digits.push_back(carry);
            carry = 0;
            ++shift;
        }
        return res;
    }

    void print() {
        for (auto it = digits.rbegin(); it != digits.rend(); ++it)
            cout << *it;
        cout << endl;
    }
};

#define SIZE 19284
void converti(int *x, int n) {
    int i = 0;
    for (; n > 0; ++i) {
        x[i] = n % 10;
        n /= 10;
    }
    for (; i < SIZE; x[i++] = 0);
}

void add_(int *x, int *y, size_t len = SIZE) {
    for (int i = 0; i < len; ++i) {
        if ((x[i] += y[i]) >= 10) {
            ++x[i+1];
            x[i] -= 10;
        }
    }
}

void mul(int *z, int *x, int *y) {
    int i, j;
    converti(z, 0);
    for (i = 0; i < SIZE; ++i)
        for (j = 0; j < x[i]; add_(z + i, y, SIZE - i), ++j);
}

void fixed_high_precision() {
    int t1, t2, n;
    cin >> t1 >> t2 >> n;
    int *v1 = new int[SIZE]; converti(v1, t1);
    int *v2 = new int[SIZE]; converti(v2, t2);
    int *pow2 = new int[SIZE];

    for (int i = 3; i <= n; ++i) {
        mul(pow2, v2, v2);
        add_(v1, pow2);
        swap(v1, v2);
    }

    int j = SIZE - 1;
    for (; j >= 0 && v2[j] != 0; --j);
    for (; j >= 0; --j)
        cout << v2[j];
    cout << endl;

    delete [] v1;
    delete [] v2;
    delete [] pow2;
}

int main() {
    int t1, t2, n;
    cin >> t1 >> t2 >> n;
    auto v1 = make_shared<HighPrecInt>(t1);
    auto v2 = make_shared<HighPrecInt>(t2);
    for (int i = 3; i <= n; ++i) {
        auto v = v1->add(v2->pow2());
        v1 = v2;
        v2 = v;
    }
    v2->print();
}

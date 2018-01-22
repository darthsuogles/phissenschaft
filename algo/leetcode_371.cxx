#include <iostream>
#include <cassert>

using namespace std;

template <size_t NBITS>
int getSum(int a, int b) {	
	int carry = 0;
	int res = 0;
	for (auto tot = NBITS; tot > 0; --tot, a >>= 1, b >>= 1) {
		res = (res << 1) ^ ((a ^ b ^ carry) & 1);
		carry = ((1 == carry) ? (a | b) : (a & b)) & 1;
	}
	int res_actual = 0;
	for (auto tot = NBITS; tot > 0; --tot, res >>= 1)
		res_actual = (res_actual << 1) ^ (res & 1);
	return res_actual;
}

int getSumGen(int a, int b) {
	if (0 == a) return b;
	if (0 == b) return a;
	int carry = 0;
	int res = 0;
	int n_pos = 0;
	for (; a != 0 || b != 0; a >>= 1, b >>= 1, ++n_pos) {
		res = (res << 1) ^ ((a ^ b ^ carry) & 1);
		if (1 == carry) {
			carry = a | b;
		} else {
			carry = a & b;
		}		
		carry = carry & 1;
	}
	if (1 == carry) {
		res = (res << 1) ^ carry; 
		++n_pos;
	}
	int res_actual = 0;
	for (; n_pos > 0; --n_pos, res >>= 1) {
		res_actual = (res_actual << 1) ^ (res & 1);
	}
	return res_actual;
}

int main() {
	for (auto i = -100; i < 100; ++i)
		for (auto j = -100; j < 100; ++j) {
			int v = getSum<32>(i, j);
			if (v != (i + j)) {
				cout << "("<< i << ", "<< j << ") v: " << v << ", " << i + j << endl;
				assert(false);
			}
		}
}

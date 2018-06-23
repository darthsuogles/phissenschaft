#include <iostream>

using namespace std;

class Solution {
    inline long get_line_intersection(int p0, int p1, int s0, int s1) {
        if (p0 > s0) {
            swap(p0, s0); swap(p1, s1);
        }
        return (p1 <= s0) ? 0 : (min(p1, s1) - s0);
    }

public:
    int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
        long lx = get_line_intersection(A, C, E, G);
        long ly = get_line_intersection(B, D, F, H);
        long ABCD = (D - B) * (C - A);
        long EFGH = (G - E) * (H - F);
        return ABCD + EFGH - lx * ly;
    }
};

Solution sol;

void TEST(int A, int B, int C, int D, int E, int F, int G, int H) {
    cout << sol.computeArea(A, B, C, D, E, F, G, H) << endl;
}

int main() {
    TEST(-3, 0, 3, 4,
         0, -1, 9, 2);

    TEST(-1500000001, 0, -1500000000, 1,
         +1500000000, 0, +1500000001, 1);
}

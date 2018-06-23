#include <iostream>

using namespace std;

int show_dist(int num) {
    // Locate the odd power just smaller than this number
    int d = 1;
    for (; (d + 2) * (d + 2) < num; d += 2);

    // Now determine which side this one resides
    int init = d * d + 1;
    // Find the initial mid point on the right-most side.
    // Then all other mid points are just `d + 1` distance away.
    int c_dist = (d + 2) / 2;

    int mid0 = c_dist + d * d;
    int mid1 = mid0 + d + 1;
    int mid2 = mid1 + d + 1;
    int mid3 = mid2 + d + 1;

    int h_dist;
    if (num <= mid0) h_dist = mid0 - num;
    else if (num <= mid1) h_dist = min(mid1 - num, num - mid0);
    else if (num <= mid2) h_dist = min(mid2 - num, num - mid1);
    else if (num <= mid3) h_dist = min(mid3 - num, num - mid2);
    else h_dist = num - mid3;

    cout << "horizontal dist: " << h_dist << endl;
    cout << "  vertical dist: " << c_dist << endl;

    return h_dist + c_dist;
}

void test(int num) {
    cout << "-------------------" << endl;
    int val = show_dist(num);
    cout << "input: " << num << " => dist: " << val << endl;
}

int main() {
    test(15);
    test(2);
    test(7);
    test(21);
    test(19);

    int num; cin >> num;
    test(num);
}

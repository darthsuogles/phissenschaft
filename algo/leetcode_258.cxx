#include <iostream>
#include <set>

using namespace std;

int digitRoot(int num) {
    // Consider the base-9 number system
    return num - 9 * ((num - 1) / 9);  // compiler: please do not optimize
}

int main() {

    cout << digitRoot(81) << endl;
}

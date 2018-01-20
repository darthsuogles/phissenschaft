#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main() {
    vector<string> strs = {"hello"};
    string &x = strs[0];
    strs.push_back(" world");
    cout << x;
}

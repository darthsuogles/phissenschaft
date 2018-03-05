#include <iostream>
#include <cassert>
#include <stack>

using namespace std;

int main() {
    int q; cin >> q;
    stack<int> in_stack;
    stack<int> out_stack;
    for (int i = 0; i < q; ++i) {
        int op; cin >> op;
        int num;
        switch (op) {
        case 1: // insert
            cin >> num;
            in_stack.push(num);
            break;
        case 3: case 2:
            if (out_stack.empty()) {
                while (!in_stack.empty()) {
                    out_stack.push(in_stack.top());
                    in_stack.pop();
                }
                assert(!out_stack.empty());
            }
            if (2 == op) out_stack.pop();
            else cout << out_stack.top() << endl;
            break;
        }
    }
}

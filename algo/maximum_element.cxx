#include <iostream>
#include <stack>

using namespace std;

int main() {
    int N;
    cin >> N;

    stack<int> val_stack;
    stack<int> max_stack;
    max_stack.push(-1);

    int query_type; int val;
    while (cin >> query_type) {
        switch (query_type) {
        case 1:
            cin >> val;
            val_stack.push(val);
            if (val >= max_stack.top()) {
                max_stack.push(val);
            }
            break;
        case 2:
            val = val_stack.top();
            if (val == max_stack.top())
                max_stack.pop();
            val_stack.pop();
            break;
        case 3:
            cout << max_stack.top() << endl;
        }
    }
}

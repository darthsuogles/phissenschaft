#include <iostream>
#include <string>
#include <cassert>
#include <stack>

using namespace std;

int main() {
    int Q; cin >> Q;
    int op;
    string text;
    string new_suffix;
    int num_chars;
    int idx;
    stack<pair<int, string>> op_stack;
    for (int i = 0; i < Q; ++i) {
        cin >> op;
        switch (op) {
        case 1: // insert
            cin >> new_suffix;
            op_stack.push(make_pair(op, new_suffix));
            text += new_suffix;
            break;
        case 2: // delete
            cin >> num_chars;
            idx = text.size() - num_chars;
            assert(idx >= 0);
            op_stack.push(make_pair(op, text.substr(idx, num_chars)));
            text.erase(text.begin() + idx, text.end());
            break;
        case 3: // show
            cin >> idx; --idx; // convert to zero-based index
            assert(idx < text.size());
            cout << text[idx] << endl;
            break;
        case 4:
            if (!op_stack.empty()) {
                int prev_op = get<0>(op_stack.top());
                string op_text = get<1>(op_stack.top());
                op_stack.pop();
                switch (prev_op) {
                case 1: // insert
                    idx = text.size() - op_text.size();
                    text.erase(text.begin() + idx, text.end());
                    break;
                case 2: // delete
                    text += op_text;
                    break;
                }
            }
            break;
        }
    }
}

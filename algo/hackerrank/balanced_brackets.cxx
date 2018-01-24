#include <iostream>
#include <string>
#include <stack>

using namespace std;

int main() {
    int N;
    cin >> N;
    string expr;
    for (int lno = 0; lno < N; ++lno) {
        cin >> expr;
        stack<char> paren_stack;
        bool circuit_break = false;
        for (char ch: expr) {
            if (circuit_break) break;
            switch (ch) {
            case '{': case '(': case '[':
                paren_stack.push(ch);
                break;
            case '}': case ')': case ']':
                if (paren_stack.empty()) {
                    circuit_break = true; break;
                }
                switch (paren_stack.top()) {
                case '{':
                    circuit_break = '}' != ch;
                    break;
                case '(':
                    circuit_break = ')' != ch;
                    break;
                case '[':
                    circuit_break = ']' != ch;
                    break;
                }
                paren_stack.pop();
                break;
            }
        }
        // Make sure that every parenthesis is accounted for.
        circuit_break = circuit_break || !paren_stack.empty();
        cout << (circuit_break ? "NO" : "YES") << endl;
    }
}

#include <iostream>
#include <string>
#include <stack>
#include <vector>
#include <cassert>
#include <boost/variant.hpp>

using namespace std;

bool is_op(char ch) {
    switch (ch) {
    case '+': case '-': case '*': case '/': return true;
    }
    return false;
}

int priority(char ch) {
    switch (ch) {
    case '*': case '/': return 2;
    case '+': case '-': return 1;
    default: return 0;
    }
}

class IsOpType: public boost::static_visitor<bool> {
public:
    bool operator()(char) const { return true; }
    bool operator()(float) const { return false; }
};

using op_or_num_type = boost::variant<float, char>;

float eval(vector<op_or_num_type> rev_polish_expr) {
    stack<float> num_stack;
    IsOpType is_op_type;
    float res;
    for (auto expr: rev_polish_expr) {
        if (boost::apply_visitor(is_op_type, expr)) {
            char op = get<char>(expr);
            float v2 = num_stack.top(); num_stack.pop();
            float v1 = num_stack.top(); num_stack.pop();
            switch (op) {
            case '+': res = v1 + v2; break;
            case '-': res = v1 - v2; break;
            case '*': res = v1 * v2; break;
            case '/': res = v1 / v2; break;
            }
        } else {
            res = get<float>(expr);
        }
        num_stack.push(res);
    }
    assert(num_stack.size() == 1);
    return num_stack.top();
}

// Shunting-yard algorithm for parsing arithmetic expressions
vector<op_or_num_type> parse_to_rev_polish(string expr) {
    stack<char> op_stack;
    vector<op_or_num_type> rev_polish_queue;

    bool has_num = false;
    float curr_num = 0.f;
    for (char ch: expr) {
        if (' ' == ch) continue;
        // Parse a number
        if ('0' <= ch && ch <= '9') {
            has_num = true;
            curr_num = curr_num * 10 + int(ch - '0');
            continue;
        }
        if (has_num) {
            rev_polish_queue.push_back(curr_num);
            has_num = false;
            curr_num = 0.f;
        }
        // Subexpression
        if ('(' == ch) {
            op_stack.push(ch);
            continue;
        } else if (')' == ch) {
            while (!op_stack.empty()) {
                char stack_op = op_stack.top();
                op_stack.pop();
                if ('(' == stack_op) break;
                rev_polish_queue.push_back(stack_op);
            }
        }
        // Operation
        if (is_op(ch)) {
            int curr_pri = priority(ch);
            while (!op_stack.empty()) {
                char stack_op = op_stack.top();
                if ('(' == stack_op || curr_pri > priority(stack_op)) break;
                rev_polish_queue.push_back(stack_op);
                op_stack.pop();
            }
            op_stack.push(ch);
            continue;
        }
    }
    if (has_num) rev_polish_queue.push_back(curr_num);
    while (!op_stack.empty()) {
        rev_polish_queue.push_back(op_stack.top());
        op_stack.pop();
    }
    return rev_polish_queue;
}


void TEST(string expr, float expected) {
    cout << "-------------" << endl;
    auto rev_polish = parse_to_rev_polish(expr);
    for (auto elem: rev_polish) cout << elem << " ";
    cout << endl;
    auto res = eval(rev_polish);
    cout << "eval " << expr << " => " << res << (res == expected ? " TRUE" : " FALSE") << endl;
}

int main() {
    TEST("1 + 2", 3);
    TEST("1 + 2 * 3", 7);
    TEST("1 + 2 * 3 * 4", 25);
    TEST("(1 + 2) * 3", 9);
    TEST("(1 + 2) / 3 * 5", 5);
    TEST("3 / 3 * 5", 5);
    TEST("(9 + 2 * (2 + 1)) / (3 * 5)", 1);
    TEST("(9 + 2 * (2 + 1)) / 3 * 5", 25);
    TEST("1 + 2 - 3", 0);
    TEST("1 + 2 + 3 * 4", 15);
    TEST("2 + 2 + 3 * 4", 16);
    TEST("2 * 2 + 3 * 4", 16);
}

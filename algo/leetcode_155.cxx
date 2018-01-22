#include <iostream>
#include <list>

using namespace std;

class MinStack {
private:
    list<long> elems;
    long min_val;
public: 
    void push(int x) {
        if (elems.empty()) {
            elems.push_front(0);            
            min_val = x;
        } else {
            long e = x - min_val;
            elems.push_front(e);
            if (e < 0) min_val = x;
        }
    }

    void pop() {
        long e = elems.front();
        if (e < 0) min_val -= e;
        elems.pop_front();
    }

    int top() {
        long e = elems.front();
        if (e > 0) {
            return (int)(e + min_val);
        } else {
            return (int)(min_val);
        }
    }
    
    int getMin() {
        return (int)(min_val);
    }
};

class MinStackDuo {
private:
    list<int> min_elems;
    list<int> ord_elems;

public:    
    void push(int x) {
        ord_elems.push_front(x);
        if (min_elems.empty() || 
            min_elems.front() >= x)
            min_elems.push_front(x);
    }
    
    void pop() {
        if (ord_elems.front() == min_elems.front())
            min_elems.pop_front();
        ord_elems.pop_front();
    }
    
    int top() {
        return ord_elems.front();
    }
    
    int getMin() {
        return min_elems.front();
    }
};


int main() {
    MinStack minStack;
    minStack.push(2);
    minStack.push(0);
    minStack.push(3);
    minStack.push(0);
    cout << minStack.getMin() << endl;   
    minStack.pop();
    cout << minStack.getMin() << endl;   
    minStack.pop();
    cout << minStack.getMin() << endl;   
    minStack.pop();
    cout << minStack.getMin() << endl;   
}

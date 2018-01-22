#include <iostream>
#include <stack>

using namespace std;

class MyQueue {
	stack<int> stq_rev;  // store the incoming
	stack<int> stq_out;  // get them out

public:
    /** Initialize your data structure here. */
    MyQueue() {
        
    }
    
    /** Push element x to the back of queue. */
    void push(int x) {
        stq_rev.push(x);
    }
    
    /** Removes the element from in front of queue and returns that element. */
    int pop() {
		if (empty()) return -1;
		int a = peek();
		stq_out.pop();
		return a;
    }
    
    /** Get the front element. */
    int peek() {
		if (stq_out.empty()) {
			while (! stq_rev.empty()) {
				int a = stq_rev.top(); 
				stq_rev.pop();
				stq_out.push(a);
			}
		}        
		if (stq_out.empty()) return -1;
		return stq_out.top();
    }
    
    /** Returns whether the queue is empty. */
    bool empty() {
        if (stq_out.empty()) 
			return stq_rev.empty();
		return false;
    }
};

/**
 * Your MyQueue object will be instantiated and called as such:
*/

int main() {
	MyQueue *obj = new MyQueue();
	for (int i = 0; i < 10; ++i) obj->push(i);
	int param_2 = obj->pop();
	cout << param_2 << endl;
	int param_3 = obj->peek();
	cout << param_3 << endl;
	bool param_4 = obj->empty();	
	cout << param_4 << endl;
}

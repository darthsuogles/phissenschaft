#include <iostream>
#include <vector>
#include <queue>
#include <utility>
#include "ListNode.hpp"

using namespace std;

using ListNode = LnkLstNode<int>;

class Solution {
    using int2 = pair<int, int>;
public:
    ListNode* mergeKLists(vector<ListNode*> &lists) {
        priority_queue<int2, vector<int2>, greater<int2>> min_heads;
        for (int i = 0; i < lists.size(); ++i) {
            auto head = lists[i];
            if (nullptr == head) continue;
            min_heads.push(make_pair(head->val, i));
        }
        ListNode *ghost = new ListNode(-1);
        ListNode *curr = ghost;
        while (!min_heads.empty()) {
            auto idx = get<1>(min_heads.top());
            min_heads.pop();
            auto node = lists[idx];
            if (nullptr == node) continue;
            curr = curr->next = node;
            if (nullptr != node->next) {
                min_heads.push(make_pair(node->next->val, idx));
            }
            lists[idx] = node->next;
        }
        curr = ghost->next;
        ghost->next = nullptr;
        delete ghost;
        return curr;
    }
};

Solution sol;

int main() {
    vector<ListNode*> lists = {
        new ListNode({1, 2, 3}),
        new ListNode({1, 3, 5}),
        new ListNode({2, 4, 6})
    };
    auto head = sol.mergeKLists(lists);
    while (nullptr != head) {
        cout << head->val << " ";
        head = head->next;
    }
    cout << endl;
}

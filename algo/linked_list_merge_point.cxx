#include <iostream>
#include <vector>
#include <memory>
#include "../ListNode.hpp"

using namespace std;
using Node = LnkLstNode1z<int>;

int FindMergeNode(shared_ptr<Node> headA, shared_ptr<Node> headB) {
    if (nullptr == headA || nullptr == headB) return -1;
    auto p = headA;
    auto q = headB;
    // They are guaranteed to meet as long as the two linked lists merge.
    while (p != q) {
        // The head changing operation will happen at most once for each head.
        if (nullptr == p) p = headB; else p = p->next;
        if (nullptr == q) q = headA; else q = q->next;
    }
    return p->val;
}

int FindMergeNodeWithCycle(shared_ptr<Node> headA, shared_ptr<Node> headB) {
    if (nullptr == headA || nullptr == headB) return -1;

    auto node = headA;
    for (; node->next; node = node->next);
    node->next = headB; // build a loop

    // Traverse the loop with two pointers with different "speed"
    auto nv1 = headA;
    auto nv2 = headA;
    while (true) {
        nv1 = nv1->next;
        nv2 = nv2->next->next;
        if (nv1 == nv2) break;
    }
    // Find the merge point by resetting one to the beginning
    // This time, the two pointers proceed with the same "speeed".
    // They are guaranteed to meet at the merging point.
    nv2 = headA;
    for (; nv1 != nv2; nv1 = nv1->next, nv2 = nv2->next);
    return nv1->val;
}

void TEST(shared_ptr<Node> headA, shared_ptr<Node> headB, int expected) {
    int res = FindMergeNode(headA, headB);
    if (res == expected) {
        cout << "PASS" << endl;
    } else {
        cout << "FAIL: expect " << expected << " but got " << res << endl;
    }
}

int main() {
    vector<int> vecA = {1, 2, 3};
    auto headA = make_shared<Node>(vecA);
    vector<int> vecB = {1};
    auto headB = make_shared<Node>(vecB);
    headB->next = headA->next;
    TEST(headA, headB, 2);
}

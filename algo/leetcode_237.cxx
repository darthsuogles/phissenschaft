#include <iostream>
#include <vector>
#include "ListNode.hpp"

using namespace std;

typedef LnkLstNode<int> ListNode;

void deleteNode(ListNode *node) {
    if (NULL == node) return;
    if (NULL == node->next) return;
    
    ListNode *curr = node, *prev = node;
    while (curr->next != NULL) {        
        prev = curr;
        curr = curr->next;
        prev->val = curr->val;
    }
    delete curr;
    prev->next = NULL;
}

void TEST(vector<int> arr, int tgt) {
    ListNode *root = new ListNode(arr); 
    root->print();                          
    ListNode *curr = root;                  
    for (; NULL != curr && curr->val != tgt; curr = curr->next);
    deleteNode(curr);
    if (NULL == curr) return;
    cout << "after deleting node " << tgt << endl;
    root->print();
}

int main() {
    
    TEST({1,2,3,4,5}, 3);
    
}

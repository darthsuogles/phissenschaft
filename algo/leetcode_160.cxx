#include <iostream>

using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x): val(x), next(NULL) {}
};

int getLen(ListNode *head) {
    int len = 0;
    for (; head != NULL; head = head->next, ++len);
    return len;
}

// Check the length and advance the longer one first
ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
    if (NULL == headA || NULL == headB) return NULL;
    int m = getLen(headA);
    int n = getLen(headB);
    ListNode *ptr1, *ptr2; int diff;
    if (m <= n) {
        diff = n-m;
        ptr1 = headA; ptr2 = headB;
    } else {
        diff = m-n;
        ptr1 = headB; ptr2 = headA;
    }
    for (int i = diff; i-- > 0; ptr2 = ptr2->next);
    for (; ptr1 && ptr2 && ptr1 != ptr2; ptr1 = ptr1->next, ptr2 = ptr2->next);
    return ptr1;
}

ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) 
{
    ListNode *p1 = headA;
    ListNode *p2 = headB;
        
    if (p1 == NULL || p2 == NULL) return NULL;

    while (p1 != NULL && p2 != NULL && p1 != p2) {
        p1 = p1->next;
        p2 = p2->next;

        //
        // Any time they collide or reach end together without colliding 
        // then return any one of the pointers.
        //
        if (p1 == p2) return p1;

        //
        // If one of them reaches the end earlier then reuse it 
        // by moving it to the beginning of other list.
        // Once both of them go through reassigning, 
        // they will be equidistant from the collision point.
        //
        if (p1 == NULL) p1 = headB;
        if (p2 == NULL) p2 = headA;
    }
        
    return p1;
}

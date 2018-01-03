"""
Split linked list into consecutive parts
"""
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

    @classmethod
    def from_list(cls, vals):
        ghost = ListNode(None)
        head = ghost
        for v in vals:
            head.next = ListNode(v)
            head = head.next
        return ghost.next

    def __repr__(self):
        head = self
        _strs = []
        while head:
            _strs.append(str(head.val))
            head = head.next
        return '=>'.join(_strs)


def splitListToParts(root, k):
    n = 0
    node = root
    while node:
        n += 1; node = node.next

    max_incr_idx = n % k
    partition_size = n // k

    res = []
    node = root
    for i in range(k):
        cnts = partition_size + int(i < max_incr_idx)
        prev = None
        curr = node
        while cnts > 0:
            if not curr: break
            prev = curr
            curr = curr.next
            cnts -= 1

        if prev: prev.next = None
        res.append(node)
        node = curr

    return res


def TEST(vals, k):
    root = ListNode.from_list(vals)
    for node in splitListToParts(root, k):
        while node:
            print(node.val, end='=>')
            node = node.next
        print()


TEST([1,2,3], 5)
TEST([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], k = 3)

"""
Copy list with random pointers
"""

class RandomListNode(object):
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None

def copyRandomList(head):
    """
    :type head: RandomListNode
    :rtype: RandomListNode
    """
    if not head: return head

    node_old2new = {}

    def get_new_node(node):
        if node is None:
            return None
        try:
            return node_old2new[node]
        except KeyError:
            new_node = RandomListNode(node.label)
            node_old2new[node] = new_node
            return new_node

    node = head
    while node is not None:
        new_node = get_new_node(node)
        new_node.next = get_new_node(node.next)
        new_node.random = get_new_node(node.random)
        node_old2new[node] = new_node
        node = node.next

    return node_old2new[head]

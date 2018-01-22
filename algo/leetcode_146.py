"""
LRU Cache
"""

class LRUCache(object):

    class DLNode(object):
        def __init__(self, key, value):
            self.key = key
            self.value = value
            self.prev = None
            self.next = None

        def remove(self):
            prev_node = self.prev
            next_node = self.next
            if prev_node is not None:
                prev_node.next = next_node
            if next_node is not None:
                next_node.prev = prev_node
            self.prev = self.next = None
            return self

        def prepend(self, node):
            if node is None: return
            prev_node = self.prev
            if prev_node is not None:
                prev_node.next = node
            self.prev = node
            node.prev = prev_node
            node.next = self

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity
        self._key_map = {}
        self._head = None

    def _get_prioritized_node(self, key):
        try:
            node = self._key_map[key]
        except KeyError:
            return None
        if node != self._head:
            self._head.prepend(node.remove())
            self._head = node
        return node

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        node = self._get_prioritized_node(key)
        return -1 if node is None else node.value

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        # If the key exists, we only have to update its value
        node = self._get_prioritized_node(key)
        if node is not None:
            node.value = value
            return

        if self._head is None:
            node = LRUCache.DLNode(key, value)
            node.prev = node
            node.next = node
        else:
            if len(self._key_map) == self.capacity:
                node = self._head.prev
                del self._key_map[node.key]
                node.key = key
                node.value = value
            else:
                node = LRUCache.DLNode(key, value)
                self._head.prepend(node)

        self._head = node
        self._key_map[key] = node


def build_cache(instructions, operands):
    assert instructions[0] == 'LRUCache'
    cache = LRUCache(operands[0][0])
    instructions.pop(0)
    operands.pop(0)
    for op, params in zip(instructions, operands):
        if 'put' == op:
            key, value = params
            print('put', key, 'value', value)
            cache.put(key, value)
        elif 'get' == op:
            key = params[0]
            print('get', key, 'value', cache.get(key))


build_cache(["LRUCache","put","put","get","get","put","get","get","get"],
            [[2],[2,1],[3,2],[3],[2],[4,3],[2],[3],[4]])

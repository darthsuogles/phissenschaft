
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

    @classmethod
    def grow(cls, arr):
        if not arr: return None
        root = TreeNode(arr[0])
        queue = [(root, 1)]
        while queue:
            node, i = queue.pop(0)
            if not node: continue

            def gen_node(j):
                try: a = arr[j-1]
                except: a = None
                if a is None: return None
                curr = TreeNode(a)
                queue.append((curr, j))
                return curr

            node.left = gen_node(2 * i)
            node.right = gen_node(2 * i + 1)

        return root


    def pprint(self):
        
        def print_level(depth, sym):
            print('|{} + {}'.format('  ' * depth, sym))

        def print_tree(root, depth):
            if not root: 
                print_level(depth, '*')
                return

            print_level(depth, root.val)
            print_tree(root.left, depth + 1)
            print_tree(root.right, depth + 1)

        print('---TREE----')
        print_tree(self, 0)


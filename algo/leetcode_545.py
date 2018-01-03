"""
Anti-clockwise outskirt
"""

from lib_arbortem import TreeNode

def get_boundary(root):
    if not root: return None
    nodes = [root]

    def add_leaf(curr):
        if not curr: return
        if curr.left is None and curr.right is None:
            nodes.append(curr)
        else:
            add_leaf(curr.left)
            add_leaf(curr.right)

    def add_left_side(curr):
        if not curr: return
        nodes.append(curr)
        if curr.left:
            add_left_side(curr.left)
            add_leaf(curr.right)
        else:
            add_left_side(curr.right)

    def add_right_side(curr):
        if not curr: return
        if curr.right:
            add_leaf(curr.left)
            add_right_side(curr.right)
        else:
            add_right_side(curr.left)
        nodes.append(curr)

    add_left_side(root.left)
    add_right_side(root.right)
    return nodes


def TEST(nodes):
    root = TreeNode.grow(nodes)
    print([n.val for n in get_boundary(root)])


TEST([1, None, 2, None, None, 3, 4])
TEST([1, 2, 3, 4, 5, 6, None, None, None, 7, 8, 9, 10])

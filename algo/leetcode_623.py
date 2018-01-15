"""
Add one layer to tree
"""

from lib_arbortem import TreeNode

def addOneRow(root, v, d):

    def attach(node, v, is_left=False):
        new_node = TreeNode(v)
        if is_left:
            new_node.left = node
        else:
            new_node.right = node
        return new_node

    def search(root, depth):
        if depth > d or root is None:
            return root
        if depth + 1 != d:
            search(root.left, depth + 1)
            search(root.right, depth + 1)
        else:
            root.left = attach(root.left, v, is_left=True)
            root.right = attach(root.right, v, is_left=False)
        return root

    if 1 == d:
        new_root = TreeNode(v)
        new_root.left = root
        return new_root
    else:
        return search(root, 1)

root = addOneRow(TreeNode.grow([4,2,6,3,1,5]), 1, 1)
root.pprint()

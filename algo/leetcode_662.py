""" Maximum level width in a binary tree
"""

from lib_arbortem import TreeNode

def widthOfBinaryTree(root):
    if root is None: return 0
    queue = [(root, 0)]
    max_width = 0
    while queue:
        w = queue[-1][1] - queue[0][1] + 1
        max_width = max(max_width, w)
        next_queue = []
        while queue:
            node, idx = queue.pop(0)
            if node.left:
                next_queue.append((node.left, 2 * idx - 1))
            if node.right:
                next_queue.append((node.right, 2 * idx))
        queue = next_queue
    return max_width

def TEST(tree_list):
    root = TreeNode.grow(tree_list)
    #root.pprint()
    print(widthOfBinaryTree(root))

TEST([1,3,None,5,3])
TEST([1,3,2,5])
TEST([1,3,2,5,None,None,9,6] + [None] * 6 + [7])

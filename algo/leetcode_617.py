""" Merge Twi Binary Trees
"""

from lib_arbortem import TreeNode

def mergeTrees(t1, t2):
    def merge(t1, t2):
        if not t1: return t2
        if not t2: return t1
        root = TreeNode(t1.val + t2.val)
        root.left = merge(t1.left, t2.left)
        root.right = merge(t1.right, t2.right)
        return root

    return merge(t1, t2)



t1 = TreeNode.grow([1,3,2,5])
t2 = TreeNode.grow([2,1,3,None,4,None,7])
t = mergeTrees(t1, t2)

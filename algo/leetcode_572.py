""" Check subtree
"""

#from lib_arbortem import TreeNode

def isSubtree(s, t):
    """
    :type s: TreeNode
    :type t: TreeNode
    :rtype: bool
    """
    if not s:
        return not t
    if not t:
        return False
    
    def same_tree(n1, n2):
        if not n1: return not n2
        if not n2: return False
        if n1.val != n2.val:
            return False
        if not same_tree(n1.left, n2.left):
            return False
        return same_tree(n1.right, n2.right)
    
    if same_tree(s, t):
        return True
    else:
        if isSubtree(s.left, t):
            return True
        return isSubtree(s.right, t)


n1 = TreeNode.grow([1,1])
n2 = TreeNode.grow([1])
print(isSubtree(n1, n2))

n1 = TreeNode.grow([3,4,5,1,2])
n2 = TreeNode.grow([4,1,2])
print(isSubtree(n1, n2))

n1 = TreeNode.grow([3,4,5,1,2,None,None,None,None,0])
n2 = TreeNode.grow([4,1,2])
print(isSubtree(n1, n2))

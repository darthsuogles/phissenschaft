""" Binary tree string repr
"""

from lib_arbortem import TreeNode

def tree2str(t):    
    def tree_repr(t):
        if not t: return ""        
        if not t.left:
            s0 = "()" if t.right else ""
        else:
            s0 = "(" + tree_repr(t.left) + ")"
        s1 = "" if not t.right else ("(" + tree_repr(t.right) + ")")
        return str(t.val) + s0 + s1

    return tree_repr(t)


t = TreeNode.grow([1,2,3,4])
print(tree2str(t))

t = TreeNode.grow([1,2,3,None,4])
print(tree2str(t))

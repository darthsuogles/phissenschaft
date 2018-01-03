""" Convert BST to greater tree
"""

from lib_arbortem import TreeNode

def convertBST(root):
    if not root: return None
    
    def convert(root, suff_val):
        """ Return min node and its converted value
        """
        _update = True
        if root.right:
            min_val, suff_val = convert(root.right, suff_val)
            if min_val == root.val:
                root.val = suff_val
                _update = False

        if _update:
            min_val = root.val
            suff_val += root.val
            root.val = suff_val

        if root.left:
            min_val, suff_val = convert(root.left, suff_val)

        return min_val, suff_val

    convert(root, 0)
    return root


root = TreeNode.grow([5,2,13])
root.pprint()
convertBST(root)
root.pprint()

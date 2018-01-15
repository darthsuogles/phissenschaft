"""
Second smallest node
"""

from lib_arbortem import TreeNode

def findSecondMinimumValue(root):
    NA = -1
    def find2e(root):
        if not root: return NA
        if (not root.left) and (not root.right):
            return NA
        lv = root.left.val
        rv = root.right.val
        val_set = set([lv, rv])
        if lv <= rv:
            curr = find2e(root.left)
            if curr != NA:
                val_set.add(curr)
        if rv <= lv:
            curr = find2e(root.right)
            if curr != NA:
                val_set.add(curr)

        ls = sorted(list(val_set))
        if len(ls) < 2: return NA
        return ls[1]
                
    return find2e(root)


def find_ref(root):
    def dfs(root, val_set):
        if not root: return
        val_set.add(root.val)
        dfs(root.left, val_set)
        dfs(root.right, val_set)
    
    val_set = set()
    dfs(root, val_set)
    ls = sorted(list(val_set))
    if len(ls) < 2: return -1
    return ls[1]

def TEST(node_list):
    root = TreeNode.grow(node_list)
    ref = find_ref(root)
    tgt = findSecondMinimumValue(root)
    assert ref == tgt, ('ref', ref, '!=', tgt)


TEST([2, 2, 5, None, None, 5, 7])
TEST([2, 2, 2])        
TEST([1,1,3,1,1,3,4,3,1,1,1,3,8,4,8,3,3,1,6,2,1])
TEST([1,1,1,None,None,1,2])

""" Lowest common ancestor
"""

from lib_arbortem import TreeNode

def lowestCommonAncestor(root, p, q):
    if not root: return None
    if isinstance(p, TreeNode):
        p = p.val
    if isinstance(q, TreeNode):
        q = q.val
    
    def lca(root, p, q):
        ret = (None, None)
        if not root:
            return ret

        if root.val == p:
            ret = (root, None)
        elif root.val == q:
            ret = (None, root)
    
        for node in [root.left, root.right]:
            rp, rq = lca(node, p, q)
            if rp: ret = (rp, ret[1])
            if rq: ret = (ret[0], rq)

            if ret[0] and ret[1]:
                anc = root if ret[0] != ret[1] else ret[0]
                return (anc, anc)

        return ret
    
    rp, rq = lca(root, p, q)
    return rp

root = TreeNode.grow([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4])
print(lowestCommonAncestor(root, 5, 1).val)

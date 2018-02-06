""" Lowest common ancestor
"""

from lib_arbortem import TreeNode

def lowestCommonAncestor(root, p, q):
    if root is None or root.val == p or root.val == q:
        return root
    anc_l = lowestCommonAncestor(root.left, p, q)
    anc_r = lowestCommonAncestor(root.right, p, q)
    if anc_l is None: return anc_r
    if anc_r is None: return anc_l
    if anc_l != anc_r: return root
    return anc_l

def lowestCommonAncestorCircuitBreak(root, p, q):
    if not root: return None

    def lca(root, p, q):
        ret = (None, None)
        if not root: return ret

        ret = (root if p == root.val else None,
               root if q == root.val else None)
        if ret[0] and ret[1]: return ret

        for node in [root.left, root.right]:
            rp, rq = lca(node, p, q)
            if rp: ret = (rp, ret[1])
            if rq: ret = (ret[0], rq)

            if ret[0] and ret[1]:
                if ret[0] != ret[1]:
                    return (root, root)
                return ret

        return ret

    rp, rq = lca(root, p, q)
    if rp != rq: return None
    return rp

root = TreeNode.grow([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4])
print(lowestCommonAncestor(root, 5, 1).val)

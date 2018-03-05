"""
House robber 3
"""

from lib_arbortem import TreeNode

def rob(root):
    if not root: return 0

    full_reward_cache = {}
    def _max_from_root(root):
        if root is None: return 0
        try: return full_reward_cache[root]
        except: pass
        val = root.val
        val += _max_without_root(root.left)
        val += _max_without_root(root.right)
        val = max(val, _max_without_root(root))
        full_reward_cache[root] = val
        return val

    partial_reward_cache = {}
    def _max_without_root(root):
        if root is None: return 0
        try: return partial_reward_cache[root]
        except: pass
        val = 0
        val += max(_max_from_root(root.left),
                   _max_without_root(root.left))
        val += max(_max_from_root(root.right),
                   _max_without_root(root.right))
        partial_reward_cache[root] = val
        return val

    return _max_from_root(root)


def TEST(vals):
    root = TreeNode.grow(vals)
    root.pprint()
    print(rob(root))


TEST([3, 2, 3, None, 3, None, 1])
TEST([4, 1, None, 2, None, None, None, 3])

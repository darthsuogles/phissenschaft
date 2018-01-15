''' Diamerter of a binary tree
'''

from lib_arbortem import TreeNode

def diameterOfBinaryTree(root):
    if not root: return 0

    def dfs(root):
        # Return: max depth and max route through @root
        if not root: return 0, 0
        depth_l, max_route_l = dfs(root.left)
        depth_r, max_route_r = dfs(root.right)
        depth = 1 + max(depth_l, depth_r)
        max_route = max(max_route_l, max_route_r)
        max_route = max(max_route, depth_l + depth_r)
        return depth, max_route

    _, max_route = dfs(root)
    return max_route


root = TreeNode.grow([1,2,3,4,5])
print(diameterOfBinaryTree(root))

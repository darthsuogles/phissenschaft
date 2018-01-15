"""
Maximum Binary Tree
"""

from lib_arbortem import TreeNode

def constructMaximumBinaryTree(nums):

    def build_tree(i, j):
        if j < i: return None

        max_val = nums[i]
        max_idx = i
        for idx in range(i+1, j+1):
            val = nums[idx]
            if val > max_val:
                max_idx = idx
                max_val = val

        root = TreeNode(max_val)
        root.left = build_tree(i, max_idx - 1)
        root.right = build_tree(max_idx + 1, j)
        return root

    return build_tree(0, len(nums) - 1)


root = constructMaximumBinaryTree([3,2,1,6,0,5])
root.pprint()

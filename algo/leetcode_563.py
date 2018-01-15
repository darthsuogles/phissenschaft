""" Binary tree tilt
""" 

from lib_arbortem import TreeNode

def findTilt(root):
    if not root: return 0

    def find_tuple(root):
        if not root:
            return 0, 0
        sum_left, tilt_left = find_tuple(root.left)
        sum_right, tilt_right = find_tuple(root.right)
        sum_root = root.val + sum_left + sum_right
        tilt_root = abs(sum_left - sum_right) + tilt_left + tilt_right
        return sum_root, tilt_root

    _, tilt = find_tuple(root)
    return tilt


root = TreeNode.grow([1, 2, 3])
root.pprint()
print(findTilt(root))

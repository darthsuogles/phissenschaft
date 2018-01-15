''' Find largest value in each tree row
'''

from lib_arbortem import TreeNode

def largestValues(root):
    if not root: return []
    level_sum = {}
    
    def search(root, level):
        if not root: 
            return
        a = root.val
        curr_max = a
        try:
            curr_max = max(level_sum[level], a)
        except: pass
        level_sum[level] = curr_max
        
        search(root.left, level + 1)
        search(root.right, level + 1)

    search(root, 0)
    return [e[1] for e in sorted(level_sum.items())]


root = TreeNode.grow([1, 3, 2, 5, 3, None, 9])
root.pprint()
res = largestValues(root)

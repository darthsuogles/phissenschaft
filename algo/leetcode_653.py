""" 2sum with binary search tree
"""

from lib_arbortem import TreeNode

def findTargetDFS(root, k):
    if root is None: return False
    def find(root, mem, k):
        if root is None: return False
        if root.val in mem: return True
        mem.add(k - root.val)
        return find(root.left, mem, k) or find(root.right, mem, k)
    return find(root, set(), k)

def findTargetFlatten(root, k):
    if root is None: return False
    def flatten(root, sorted_buf):
        if root is None: return
        flatten(root.left, sorted_buf)
        sorted_buf.append(root.val)
        flatten(root.right, sorted_buf)

    arr = []
    flatten(root, arr)
    i = 0; j = len(arr) - 1
    while i < j:
        curr = arr[i] + arr[j]
        if curr == k:
            return True
        if curr < k:
            i += 1
        else:
            j -= 1

    return False



root = TreeNode.grow([5,3,6,2,4,None,7])
print(findTarget(root, 9))
print(findTarget(root, 28))

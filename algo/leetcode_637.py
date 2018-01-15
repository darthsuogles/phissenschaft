"""
Binary tree level average
"""

from lib_arbortem import TreeNode

def averageOfLevel(root):
    avgs = []
    if not root: return avgs

    queue = [root]
    while queue:
        next_queue = []
        avg = 0.0
        cnts = 0
        for node in queue:
            if node.left:
                next_queue.append(node.left)
            if node.right:
                next_queue.append(node.right)
            avg += node.val
            cnts += 1
        avg = avg / cnts
        avgs.append(avg)
        del queue
        queue = next_queue

    return avgs


def TEST(root):
    print(averageOfLevel(root))

root = TreeNode.grow([3, 9, 20, None, None, 15, 7])
TEST(root)

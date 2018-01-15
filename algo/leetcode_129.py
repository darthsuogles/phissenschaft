""" sum root-to-leaf numbers
""" 

from lib_arbortem import TreeNode

def sumNumbers(root):
    def kern(root, prefix=0):
        if not root: return 0    
        val = 10 * prefix + root.val
        if not root.left and not root.right:
            return val
        left_sum = kern(root.left, val)
        right_sum = kern(root.right, val)
        return left_sum + right_sum

    return kern(root)


def TEST(root, tgt):
    root.pprint()
    val = sumNumbers(root)
    if val != tgt:
        print('Error', val, 'but expect', tgt)
    else:
        print('Ok')

TEST(TreeNode.grow([1,2,3]), 25)
    

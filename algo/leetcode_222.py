

def countNodes(root):
    if root is None:
        return 0

    def depth(node):
        if node is None:
            return 0
        return 1 + max(depth(node.left), depth(node.right))

    
    d_left = depth(root.left)
    d_right = depth(root.right)
    if d_left == d_right:
        return 1 + 2 * pow(2, d_left) 
    
    return 1 + countNodes(root->left) + countNodes(root->right)


def countNodes(root):
    """
    :type root: TreeNode
    :rtype: int
    """
    if root is None:
        return 0

    # Only need to check if the far left and far right equal
    nl = root.left; dl = 0
    while nl: 
        nl = nl.left; dl += 1
    nr = root.right; dr = 0
    while nr:
        nr = nr.right; dr += 1

    if dl == dr:
        return pow(2, (dl+1)) - 1

    return 1 + countNodes(root.left) + countNodes(root.right)

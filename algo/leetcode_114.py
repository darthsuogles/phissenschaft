''' Flatten binary tree
'''

from lib_arbortem import TreeNode

def flattenPtr(root):
    ''' Keep pointers to both first and last node
    '''
    if not root: return None
    
    def flatten_bidirect(root):
        last = root
        left_sub, right_sub = root.left, root.right
        if left_sub:
            left_first, left_last = flatten_bidirect(left_sub)
            root.right = left_first
            last = left_last
        if right_sub:            
            right_first, right_last = flatten_bidirect(right_sub)
            last.right = right_first
            last = right_last
            
        root.left = None
        return root, last

    root_iter, _ = flatten_bidirect(root)
    return root_iter


def flattenMorris(root):
    ''' Using Morris traversal to flatten the tree
    '''
    curr = root
    while curr:
        if curr.left:
            prev = curr.left
            while prev.right:
                prev = prev.right
                
            prev.right = curr.right
            curr.right = curr.left
            tmp = curr.left; curr.left = None
            curr = tmp
        else:
            curr = curr.right


def morrisPreOrd(root):
    ''' Morris traversal, no recursive nor stack
        - Create a loop from left subtree to root
        - When the loop is detected, remove it
    '''     
    curr = root
    while curr:
        if curr.left:
            prev = curr.left
            while prev.right and prev.right != curr:
                prev = prev.right
                
            if not prev.right:
                prev.right = curr
                print(curr.val)  # pre-order traversal
                curr = curr.left
                continue

            if prev.right == curr:
                prev.right = None

        else:
            print(curr.val)  # pre-order traversal

        curr = curr.right        
        


def flatten(root):
    if not root: return
    
    def flatten_kern(root, head=None):
        if not root: return head
        head = flatten_kern(root.right, head)
        head = flatten_kern(root.left, head)
        root.left = None
        root.right = head
        return root

    head = flatten_kern(root, None)
    return head


def TEST(arr):
    root = TreeNode.grow(arr)
    root.pprint()
    root_iter = flatten(root)
    while root_iter:
        print(root_iter.val, end='\t')
        root_iter = root_iter.right


root = TreeNode.grow([1,2,None,3,5])
flattenMorris(root)
while root:
    print(root.val, end='\t')
    root = root.right

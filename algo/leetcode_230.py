''' K-th smallest element in binary search tree
''' 

from lib_arbortem import TreeNode

def kthSmallest(root, k):
    ''' Would be better if we can preprocess it
        by storing the size of each subtree
    
        Or we can use iterator (generator comprehension)
    ''' 
    if root is None: return None
        
    def search(root, k):
        ''' Return
            1. the k-th element or None
            2. size of subtree 
        '''
        if root is None:
            return None, 0        
        
        node, size = search(root.left, k)
        tot_size = size
        res = node
            
        # 1-based indexing
        if tot_size == k - 1:
            res = root
        tot_size += 1

        node, size = search(root.right, k - tot_size)
        tot_size += size
        if not res:
            res = node
                    
        return res, tot_size

    res, _ = search(root, k)
    assert res
    return res.val



root = TreeNode.grow([3, 1, 4, None, 2])
root.pprint()

    
print(kthSmallest(root, 2))
print(kthSmallest(root, 0))
print(kthSmallest(root, 3))

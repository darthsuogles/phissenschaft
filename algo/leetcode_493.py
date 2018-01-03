''' Reverse pairs
'''

class BSTNode(object):
    ''' Simple binary search tree without balancing
    '''
    def __init__(self, a):
        self.val = a
        self.left = None
        self.right = None
        self.size = 1  # number of nodes in the subtree


    def get_insert_index(self, a):
        ''' Return the linear index 
        '''        
        if self.val == a:
            if self.right:
                return self.size - self.right.size
            else:
                return self.size

        if a < self.val:
            if not self.left:
                return 0
            return self.left.get_insert_index(a)
                
        if a > self.val:
            if not self.right:
                return self.size
            num_nodes_left = self.size - self.right.size 
            return num_nodes_left + self.right.get_insert_index(a)


    def insert(self, a):
        self.size += 1

        if self.val == a:
            return

        if a < self.val:
            if self.left:
                self.left.insert(a)
            else:
                self.left = BSTNode(a)
            return

        if a > self.val:
            if self.right:
                self.right.insert(a)
            else:
                self.right = BSTNode(a)
            return


def reversePairsBST(nums):
    if not nums: return 0

    root = BSTNode(nums[0])
    pair_cnts = 0
    for a in nums[1:]:
        k = root.get_insert_index(2 * a)
        pair_cnts += root.size - k
        root.insert(a)

    return pair_cnts


def reversePairs(nums):
    """ Maintain a sorted array and find insertion locations
    """
    if not nums: return 0
    from bisect import bisect_left, bisect_right
    prefix_sorted = []
    counts = 0
    for a in nums:
        # Find all elements <= 2 * a
        i = bisect_right(prefix_sorted, 2 * a)
        counts += len(prefix_sorted) - i
        # Maintain the sorted prefix elements
        j = bisect_left(prefix_sorted, a)
        prefix_sorted.insert(j, a)

    return counts

def TEST(nums, tgt):
    res = reversePairs(nums)
    if res != tgt:
        print('Error', res, 'but expect', tgt)
    else:
        print('Ok')

TEST([1, 3, 2, 3, 1], 2)
TEST([2, 4, 3, 5, 1], 3)

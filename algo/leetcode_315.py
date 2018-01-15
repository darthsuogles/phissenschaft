''' Counting con/dis-cordant pairs
'''

def getClosestSmaller(nums):
    ''' Get the closest smaller element on the right
    '''
    if not nums: return []
    n = len(nums)
    stq = []
    bnd_right_idx = [-1] * n
    # Use a stack to track the next larger number
    for i, a in enumerate(nums):
        while stq:
            j = stq[-1]
            b = nums[j]
            if b <= a:
                break
            bnd_right_idx[j] = i
            stq = stq[:-1]

        stq.append(i)

    print('stack', stq)
    print('right bound', bnd_right_idx)
    return [nums[j] if -1 != j else None
            for j in bnd_right_idx]


class BSTNode(object):
    def __init__(self, a):
        self.val = a
        self.left = None
        self.right = None
        self.size = 1
    
    def insert(self, a):
        if self.val == a:
            self.size += 1
            if self.left:
                return self.left.size
            else:
                return 0
            
        if a < self.val:
            self.size += 1
            if self.left:
                return self.left.insert(a)
            else:
                self.left = BSTNode(a)
                return 0

        if a > self.val:
            if self.right:
                cnt_smaller = self.size - self.right.size
                cnt = cnt_smaller + self.right.insert(a)
            else:
                self.right = BSTNode(a)
                cnt = self.size

            self.size += 1
            return cnt


def countSmallerBinarySearch(nums):
    ''' A modified binary search tree where insertion 
        keeps track of the number of elements smaller
    '''
    if not nums: return []
    n = len(nums)
    if 1 == n: return [0]

    root = BSTNode(nums[-1])
    res = [0] * n
    for i in range(n-2, -1, -1):
        a = nums[i]
        res[i] = root.insert(a)

    return res

# Using a modified merge sort
def count_smaller_aux(nums, idx):
    if nums is None: return [], [], []
    n = len(nums)
    if 0 == n: return [], [], []
    if 1 == n: return [0], nums, idx

    # The right 
    hf = n // 2
    res_lo, sorted_lo, idx_lo = count_smaller_aux(nums[:hf], idx[:hf])
    res_hi, sorted_hi, idx_hi = count_smaller_aux(nums[hf:], idx[hf:])

    i = 0; j = 0
    res_buf = []; idx_buf = []; sorted_buf = []
    while i < len(sorted_lo) and j < len(sorted_hi):
        a = sorted_lo[i]; b = sorted_hi[j]
        if a > b:
            sorted_buf += [b]
            idx_buf += [idx_hi[j]]
            res_buf += [res_hi[j]]
            j += 1
        else:
            sorted_buf += [a]
            idx_buf += [idx_lo[i]]
            res_buf += [res_lo[i] + j]
            i += 1                
            
    cnt_hi = n - hf
    while i < len(sorted_lo):
        sorted_buf += [sorted_lo[i]]
        idx_buf += [idx_lo[i]]
        res_buf += [res_lo[i] + cnt_hi]
        i += 1

    while j < len(sorted_hi):
        sorted_buf += [sorted_hi[j]]
        idx_buf += [idx_hi[j]]
        res_buf += [res_hi[j]]
        j += 1
    
    return res_buf, sorted_buf, idx_buf


def countSmaller(nums):
    idx = list(range(len(nums)))
    res_map, sorted_nums, res_idx = count_smaller_aux(nums, idx)
    res = [0] * len(nums)
    for i in idx:
        res[res_idx[i]] = res_map[i]
    return res


def TEST(nums, tgt):
    res = countSmaller(nums)
    if res != tgt:
        print("Error", res, "but expect", tgt)
    else:
        print("Ok")

TEST([5,2,6,1], [2,1,1,0])
TEST([], [])
TEST([1], [0])
TEST([100,26,78,27,100,33], [4, 0, 2, 0, 1, 0])

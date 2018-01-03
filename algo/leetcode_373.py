''' Find K pairs smallest sum
'''

from copy import deepcopy
import itertools
import heapq

def kSmallestPairs(nums1, nums2, k):
    if not nums1 or not nums2: return []
    hpq = []
    heapq.heappush(hpq, [nums1[0] + nums2[0], 0, 0])
    res = []
    while hpq and len(res) < k:
        _, i, j = heapq.heappop(hpq)
        res.append([nums1[i], nums2[j]])
        if j + 1 < len(nums2):
            heapq.heappush(hpq, [nums1[i] + nums2[j+1], i, j + 1])
        if 0 == j:
            if i + 1 < len(nums1):
                heapq.heappush(hpq, [nums1[i+1] + nums2[0], i + 1, 0])
        
    return res


def kSmallestRef(nums_list, k):

    def sub_stream(ls):
        if 1 == len(ls):
            return map(lambda u: [u, u], ls[0][:k])

        streams = map(lambda vp: 
                      ([u + vp[0]] + [u] + vp[1:] for u in ls[0]),
                      sub_stream(ls[1:]))

        stream_merged = heapq.merge(*streams)
        return itertools.islice(stream_merged, k)
    
    return [e[1:] for e in sub_stream(nums_list)]


def kSmallestTuples(nums_list, k):

    def k_small_kern(nums_list, k):
        nums_list = [v for v in nums_list if v]
        if not nums_list: return []
        n = len(nums_list)

        if 1 == n:  # one list only, return top-k
            return [[u, u] for u in nums_list[0][:k]]

        ax = nums_list[0]
        bx = k_small_kern(nums_list[1:], k)
        hpq = []
        res = []
        m = len(ax); n = len(bx)
        def push(i, j):
            if not (i < m and j < n): return
            el = [ax[i] + bx[j][0]] + [(i, j)] + [ax[i]] + bx[j][1:]
            heapq.heappush(hpq, el)

        def pop():
            if not hpq: return None
            el = heapq.heappop(hpq)
            i, j = el[1]
            vals = el[2:]
            v_sum = el[0]
            return i, j, v_sum, vals

        push(0, 0)
        while hpq and len(res) < k:
            i, j, v_sum, vals = pop()
            res.append([v_sum] + vals)
            if j + 1 < n: 
                push(i, j + 1)
            if 0 == j:
                if i + 1 < m:
                    push(i + 1, 0)

        return res

    return [v[1:] for v in k_small_kern(nums_list, k)]


def TEST2(nums1, nums2, k):
    res = kSmallestPairs(nums1, nums2, k)
    for v in res:
        print(v)
    print('------------------')
        
def TESTN(nums_list, k):
    res = kSmallestTuples(nums_list, k)
    for v in res:
        print(v)
    print('------------------')

TEST2([1,1,2], [1,2,3], 10)
# TEST2([1,2,4,5,6], [3,5,7,9], 3)
# TEST2([-10,-4,0,0,6],
#       [3,5,6,7,8,100],
#       10)
TESTN([[1,1,2], [1,2,3], [-1,0,2]], 10)

"""
Find K-th smallest pair distance
"""

def smallestDistancePairRef2(nums, k):
    import heapq
    from collections import Counter, defaultdict

    elems_cntr = Counter(nums)
    for _, cnts in elems_cntr.items():
        k -= cnts * (cnts - 1) // 2
        if k <= 0: return 0

    nums = sorted(elems_cntr.keys())

    prev = nums.pop(0)
    prev_dists_cnts = defaultdict(int)

    min_val = 1 << 32
    dist_coll = defaultdict(int)
    for curr in nums:
        n = elems_cntr[curr]
        diff = curr - prev
        next_dists_cnts = defaultdict(int)
        prev_dists_cnts[0] += elems_cntr[prev]
        for dist, cnts in prev_dists_cnts.items():
            val = dist + diff
            if val > min_val:
                continue
            next_dists_cnts[val] += cnts
            dist_coll[val] += cnts * n

        k0 = k
        coll0 = sorted(dist_coll)
        i = 0
        while i < len(coll0):
            a = coll0[i]
            i += 1
            k0 -= dist_coll[a]
            if k0 <= 0:
                max_val = a; break

        while i < len(coll0):
            del dist_coll[coll0[i]]
            i += 1

        prev = curr
        prev_dists_cnts = next_dists_cnts

    return sorted(dist_coll)[-1]


def smallestDistancePairRef(nums, k):
    import heapq
    dists = []
    for i, a in enumerate(nums):
        for j in range(i+1, len(nums)):
            b = nums[j]
            heapq.heappush(dists, -abs(a - b))
            if len(dists) > k:
                heapq.heappop(dists)

    return -sorted(dists)[0]


def smallestDistancePair(nums, k):
    nums.sort()

    def has_k_smaller_distances(guess):
        """ Are there k or more pairs with distance <= guess? """
        count = left = 0
        for right, x in enumerate(nums):
            while x - nums[left] > guess:
                left += 1
            count += right - left
        return count >= k

    # At least one "F" and one "T"
    # This is a step function: e.g. "FFFFFTTTTTTT"
    lo = 0
    hi = nums[-1] - nums[0]
    while lo < hi:
        # If there are only two elements, mi == lo
        # Also, mi + 1 <= hi
        mi = (lo + hi) // 2
        # If the subarray is always "F", lo will stay
        if has_k_smaller_distances(mi):
            hi = mi  # hi will always point to a "T" position
        else:
            lo = mi + 1  # lo will not point to a "F" position

    # When outside the loop, lo == hi
    # Also, lo - 1 points to an "F" position
    # Thus lo always points to the first "T" position

    return lo

def TEST(nums, k):
    ref = smallestDistancePairRef(nums, k)
    tgt = smallestDistancePair(nums, k)
    print('tgt', tgt, 'ref', ref)


TEST([1,3,1], 1)
TEST([62,100,4], 2)
TEST([9,10,7,10,6,1,5,4,9,8], 18)

"""
Muscle memory: range minimum query
"""

import math

class RangeMinQuery(object):
    def __init__(self, nums):
        self._nums = nums
        self._preproc()

    def _preproc(self):
        n = len(self._nums)
        self._block_size = int(math.ceil(math.sqrt(n)))
        assert self._block_size > 0
        self._blocked_mins = []
        for begin_idx in range(0, n, self._block_size):
            bound_idx = min(begin_idx + self._block_size, n)
            min_idx = begin_idx
            min_val = self._nums[min_idx]
            for i in range(begin_idx, bound_idx):
                if self._nums[i] < min_val:
                    min_val = self._nums[i]
                    min_idx = i
            self._blocked_mins.append(min_idx)

    def _find_min_in_nums(self, i, j, ref_min_idx=None):
        min_idx = i
        min_val = self._nums[min_idx]
        for k in range(i, j + 1):
            if self._nums[k] < min_val:
                min_val = self._nums[k]
                min_idx = k

        if ref_min_idx is not None and self._nums[ref_min_idx] < min_val:
            return ref_min_idx
        else:
            return min_idx

    def find(self, i, j):
        assert i <= j and 0 <= i and j < len(self._nums)

        lo_block_idx = i // self._block_size
        hi_block_idx = j // self._block_size

        # The range indices reside in the same block
        if lo_block_idx == hi_block_idx:
            return self._find_min_in_nums(i, j)

        lo_block_i1 = min(len(self._nums),
                          (lo_block_idx + 1) * self._block_size)
        min_idx = self._find_min_in_nums(i, lo_block_i1)
        min_val = self._nums[min_idx]

        hi_block_j0 = hi_block_idx * self._block_size
        min_idx = self._find_min_in_nums(hi_block_j0, j, min_idx)
        min_val = self._nums[min_idx]

        for block_idx in range(lo_block_idx + 1, hi_block_idx):
            min_idx_in_block = self._blocked_mins[block_idx]
            min_block_val = self._nums[min_idx_in_block]
            if min_block_val < min_val:
                min_val = min_block_val
                min_idx = min_idx_in_block

        return min_idx


rmq = RangeMinQuery([2, 4, 3, 1, 6, 7, 8, 9, 1, 7])

print(rmq.find(2, 7))
print(rmq.find(1, 1))

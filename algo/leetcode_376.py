""" Wiggle subsequence
"""

def wiggleMaxLength(nums):
    n = len(nums)
    if n < 2: return n
    # Find smallest and largest from left
    wiggle_pos = [1]
    wiggle_neg = [1]
    n = len(nums)
    max_len = 0
    for i in range(1, n):
        a = nums[i]
        if a == nums[i-1]:
            _pos_len = wiggle_pos[i-1]
            _neg_len = wiggle_neg[i-1]
        else:
            _pos_len = 0
            _neg_len = 0
            for j in range(i-1, -1, -1):
                b = nums[j]
                if a < b:
                    _neg_len = max(_neg_len, 1 + wiggle_pos[j])
                if a > b:
                    _pos_len = max(_pos_len, 1 + wiggle_neg[j])

        wiggle_pos += [_pos_len]
        wiggle_neg += [_neg_len]
        max_len = max(max_len, max(_pos_len, _neg_len))

    return max_len


print(wiggleMaxLength([1,7,4,9,2,5]))
print(wiggleMaxLength([1,17,5,10,13,15,10,5,16,8]))
print(wiggleMaxLength([1,2,3,4,5,6,7,8,9]))
print(wiggleMaxLength([2,2]))

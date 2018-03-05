"""
Assigning array elements to processors
"""

n = 54
p = 7
nums = range(n)  # the data array
r = n % p

# Elements controlled by processors
blocks = [None] * p
for i in range(p):
    j0 = i * (n // p) + min(i, r)
    j1 = (i + 1) * (n // p) + min(i + 1, r)
    blocks[i] = set(nums[j0:j1])
    print(blocks[i], j1 - j0)

# Finding elements' controlling processor
for j in nums:
    batch = n // p
    i = max(j // (batch + 1), (j - r) // batch)
    assert j in blocks[i]

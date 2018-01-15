""" Heap ops

Ref: https://docs.python.org/3.6/library/heapq.html
"""

import heapq

arr = [1,2,5,4,2,3,4]

# Comparison on the first element of the tuple
hpq = []
for idx, a in enumerate(arr):
    heapq.heappush(hpq, (a, idx))

print('------ Heap Content ------')
while hpq:
    print(heapq.heappop(hpq))

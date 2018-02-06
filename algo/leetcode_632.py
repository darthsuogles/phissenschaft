"""
Smallest range
"""

def smallestRange(nums):
    if not nums: return []
    import heapq
    from math import inf

    # Stores all the numbers in the head
    head_range = []
    max_val = -inf
    for idx, lst in enumerate(nums):
        assert lst is not None
        max_val = max(max_val, lst[0])
        heapq.heappush(head_range, (lst[0], idx))
        lst.pop(0)

    best_range = (-inf, inf)
    while head_range:
        curr_min, which_list = heapq.heappop(head_range)
        if max_val - curr_min < best_range[1] - best_range[0]:
            best_range = (curr_min, max_val)
        lst = nums[which_list]
        # When one of the list is empty, we will have to quit,
        # as we won't be getting more items from there.
        if not lst: break
        next_val = lst[0]; lst.pop(0)
        max_val = max(max_val, next_val)
        heapq.heappush(head_range, (next_val, which_list))

    return best_range


def TEST(nums):
    print(smallestRange(nums))


TEST([[4,10,15,24,26], [0,9,12,20], [5,18,22,30]])

""" Queue reconstruction by height
"""

def reconstructQueue(people):
    if not people: return []
    # Order: desc by values, asc by prefix counts
    # Every new insertion is sure that everything in the
    # list is greater than or equal to itself. 
    # Thus we only have to insert to its corresponding 
    # location denoted by the prefix count
    arr = sorted(people, key=lambda p: (-p[0], p[1]))
    out_list = []
    for h, k in arr:
        out_list.insert(k, [h, k])
    return out_list

def reconstructQueueStq(people):
    from bisect import bisect_right

    p_dict = set([])
    pref_sum = {}
    stack = []

    for h, k in people:
        pref_sum[h] = 0
        p_dict.add((h, k))
        if 0 == k:
            stack.append((h, k))

    h_asc = sorted(list(pref_sum.keys()))
    stack = sorted(stack, key=lambda p: -p[0])
    out_list = []

    while stack:
        h0, k0 = stack.pop(-1)
        out_list.append([h0, k0])
        i_bnd = bisect_right(h_asc, h0)
        for i in range(i_bnd-1, -1, -1):
            h = h_asc[i]
            k = pref_sum[h] + 1
            pref_sum[h] = k
            if (h, k) in p_dict:
                stack.append((h, k))

    return out_list


people = [[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]
print(reconstructQueue(people))

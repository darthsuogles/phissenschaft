''' Beautiful Arrangement
'''

def countArrangement(N):
    from collections import defaultdict
    pos_avail = defaultdict(list)
    for a in range(1, N + 1):
        for i in range(1, N + 1):
            if 0 == a % i or 0 == i % a:
                pos_avail[a].append(i)

    def search(a, pos_avail, slot_used):
        if a > N:
            return int(all(slot_used))

        cnt = 0
        for i in pos_avail[a]:
            if slot_used[i]: 
                continue
            slot_used[i] = True
            cnt += search(a + 1, pos_avail, slot_used)
            slot_used[i] = False
        return cnt
    
    slot_used = [False] * (N + 1)
    slot_used[0] = True
    return search(1, pos_avail, slot_used)


print(countArrangement(2))
print(countArrangement(15))

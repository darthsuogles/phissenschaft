"""
Signal propagation time in a network
"""

def networkDelayTime(times, N, K):
    if not times: return -1

    # Build linked list
    from collections import defaultdict
    neighbors = defaultdict(list)
    for u, v, dt in times:
        neighbors[u].append((v, dt))

    import heapq
    hpq = [(0, K)]
    visited = set([])
    max_dt = 0
    while hpq:
        elapsed, u = heapq.heappop(hpq)
        if u in visited: continue
        visited.add(u)
        max_dt = max(max_dt, elapsed)
        for v, dt in neighbors[u]:
            if v in visited: continue
            heapq.heappush(hpq, (elapsed + dt, v))

    if len(visited) < N:
        return -1
    return max_dt


def TEST(times, N, K):
    print(networkDelayTime(times, N, K))


TEST([[1,2,3], [2,3,2], [3,4,1]], 4, 1)
TEST([[2,1,5], [2,3,2], [3,4,1]], 4, 2)
TEST([[1,2,1],[2,3,2],[1,3,2]], 3, 1)

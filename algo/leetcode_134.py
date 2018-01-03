"""
Traveling between gas stations
"""

def canCompleteCircuit(gas, cost):
    if not gas or not cost: return 0
    idx, tot, nondim_sum = 0, 0, 0
    # Takes the longest running non-diminishing sub-array (till end)
    for i, (g, c) in enumerate(zip(gas, cost)):
        nondim_sum += g - c
        if nondim_sum < 0:
            idx = i + 1
            tot += nondim_sum
            nondim_sum = 0
    tot += nondim_sum
    # If the total amount is >= 0, then the corresponding prefix
    # must have a smaller |abs| value. Otherwise the current running
    # non-diminishing postfix sub-array is not the longest one.
    return idx if tot >= 0 else -1

def canCompleteCircuitRef(gas, cost):
    if not gas or not cost: return 0

    N = len(gas)
    excl_pref_sums = [0] * (N+1)
    excl_pref_mins = [0] * (N+1)
    pref_tot = 0
    pref_min = 0
    for i in range(N):
        gas_diff = gas[i] - cost[i]
        if pref_tot < pref_min:
            pref_min = pref_tot
        excl_pref_sums[i] = pref_tot
        excl_pref_mins[i] = pref_min
        pref_tot += gas_diff
    excl_pref_sums[N] = pref_tot

    post_tot = 0
    post_max = -cost[N-1]
    incl_post_mins = [0] * N
    for i in range(N-1, -1, -1):
        post_tot += gas[i] - cost[i]
        if post_tot > post_max:
            post_max = post_tot
        curr_post_sum = pref_tot - excl_pref_sums[i]
        # lowest point of gas from i ... N-1
        incl_post_mins[i] = curr_post_sum - post_max

    for i in range(N):
        if incl_post_mins[i] < 0:
            continue
        curr_post_rem = pref_tot - excl_pref_sums[i]
        if curr_post_rem + excl_pref_mins[i] >= 0:
            return i
    return -1


def TEST(gas, cost):
    tgt = canCompleteCircuit(gas, cost)
    ref = canCompleteCircuitRef(gas, cost)
    assert tgt == ref, (tgt, ref)

print('------TEST-CASES-------')
TEST([5], [4])
TEST([0,3,2], [1,3,2])
TEST([1,2], [2,1])

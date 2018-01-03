"""
Baseball game
"""

def callPoints(ops):
    if not ops: return 0
    last_valids = []
    tot_pts = 0
    # ops_idx = 1
    # pts_idx = 1
    for op in ops:
        curr = None
        if "C" == op:
            tot_pts -= last_valids[-1]
            last_valids = last_valids[:-1]
            # msg = "Operation {}: round {}'s data was invalid, the sum is {}"
            # print(msg.format(ops_idx, pts_idx, tot_pts))
            # ops_idx += 1
        elif "D" == op:
            curr = 2 * last_valids[-1]
            tot_pts += curr
            # msg = "Round {}: get {} points, the sum is {}"
            # print(msg.format(pts_idx, curr, tot_pts))
            # pts_idx += 1
        elif "+" == op:
            p1, p2 = last_valids[-2:]
            curr = p1 + p2
            tot_pts += curr
            # msg = "Round {}: get {} + {} = {} points, the sum is {}"
            # print(msg.format(pts_idx, p1, p2, curr, tot_pts))
            # pts_idx += 1
        else:
            curr = int(op)
            tot_pts += curr
            # msg = "Round {}: get {} points, the sum is {}"
            # print(msg.format(pts_idx, curr, tot_pts))
            # pts_idx += 1

        if curr is not None:
            last_valids.append(curr)

    return tot_pts


print('----TEST------')
callPoints(["5","2","C","D","+"])
print('----TEST------')
callPoints(["5","-2","4","C","D","9","+","+"])

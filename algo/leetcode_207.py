''' Course schedule
'''

def findOrder(numCourses, prerequisites):
    from collections import defaultdict
    tbl_locks = defaultdict(list)
    tbl_num_prereqs = defaultdict(lambda: 0)
    for pres in prerequisites:
        curr, deps = pres[0], pres[1:]        
        tbl_num_prereqs[curr] += len(deps)
        for c in deps:
            tbl_locks[c] += [curr]
        
    res = []
    stq = [c for c in range(numCourses) if tbl_num_prereqs[c] == 0]
    while stq:
        curr = stq[0]; stq = stq[1:]
        res += [curr]
        for c in tbl_locks[curr]:
            c_num_pres = tbl_num_prereqs[c] - 1
            if 0 == c_num_pres:
                stq += [c]
            tbl_num_prereqs[c] = c_num_pres

    if len(res) < numCourses:
        return []
    return res


def TEST(n, pres):
    print(findOrder(n, pres))


# TEST(2, [[1,0]])
# TEST(2, [[1,0], [0,1]])        
# TEST(4, [[1,0],[2,1],[3,2],[1,3]])
TEST(4, [[1,0],[2,0],[3,1],[3,2]])
TEST(4, [[1,0],[2,1],[3,2],[1,3]])

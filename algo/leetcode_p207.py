class Solution:
    # @param {integer} numCourses
    # @param {integer[][]} prerequisites
    # @return {boolean}
    def canFinish(self, numCourses, prerequisites):
        graph_deps = {}
        num_preqs = [0] * numCourses
        for u, v in prerequisites:
            try:
                if v not in graph_deps[u]:
                    graph_deps[u].add(v)
                    num_preqs[v] += 1
            except KeyError:
                graph_deps[u] = set([v])
                num_preqs[v] += 1
                pass

        cour_q = [u for u in range(numCourses) if num_preqs[u] == 0]
        num_unlocked = len(cour_q)
        while len(cour_q) > 0:
            cour = cour_q[0]; cour_q = cour_q[1:]            
            if cour not in graph_deps:
                continue            
            for v in graph_deps[cour]:
                num_preqs[v] -= 1
                if num_preqs[v] == 0:
                    num_unlocked += 1
                    cour_q += [v];

        return (num_unlocked == numCourses)


    
            
sol = Solution()
print(sol.canFinish(2, [[1,0]]))
print(sol.canFinish(2, [[1,0],[0,1]]))

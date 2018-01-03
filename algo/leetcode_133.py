''' Clone graph
'''

class UndirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []

def cloneGraph(node):
    ''' Breadth first search solution
    '''
    if not node: return node

    new_nodes = {node: UndirectedGraphNode(node.label)}
    queue = [node]
    while queue:
        curr = queue[0]; queue = queue[1:]
        replica = new_nodes[curr]
        for next in curr.neighbors:
            if next in new_nodes: 
                replica.neighbors.append(new_nodes[next])
                continue

            next_replica = UndirectedGraphNode(next.label)
            replica.neighbors.append(next_replica)
            queue.append(next)

    return new_nodes[node]
    

def cloneGraph(node):
    ''' The DFS solution
    '''
    if not node: return node

    new_nodes = {}
    def dfs_clone(curr):
        curr_new = UndirectedGraphNode(curr.label)
        new_nodes[curr] = curr_new
        for next_old in curr.neighbors:
            try:
                next_new = new_nodes[next_old]
            except: 
                next_new = dfs_clone(next_old)

            curr_new.neighbors.append(next_new)

        return curr_new

    return dfs_clone(node)


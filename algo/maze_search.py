''' Searching in a maze
'''

def find_all_paths(maze):
    if not maze: return False
    m = len(maze)
    n = len(maze[0])
    if 0 == n: return False

    # Find starting point and goals
    s = None; t = None
    goals = set()
    for i in range(m):
        for j in range(n):
            ch = maze[i][j]
            if 'A' == ch:
                s = i; t = j
            elif 'G' == ch:
                goals.add((i, j))    

    # Find paths to goals 
    def dfs(s, t, visited, goals):
        for i in range(max(0, s-1), 
                       min(s+1, m-1) + 1):
            for j in range(max(0, t-1), 
                           min(t+1, n-1) + 1):
                if (i, j) in visited:
                    continue

                if 'G' == maze[i][j]:
                    goals.add((i, j))
                    visited.add((i, j))
                    dfs(i, j, visited, goals)
                  
                elif '0' == maze[i][j]:
                    visited.add((i, j))
                    dfs(i, j, visited, goals)
                    
    visited = set()
    goals_found = set()
    dfs(s, t, visited, goals_found)
    print('found', goals_found, 'out of', goals)


maze = [
    '000111',
    'A101G1',
    '10G011',
    '100101'
]

find_all_paths(maze)

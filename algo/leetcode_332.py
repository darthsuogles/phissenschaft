''' Itinerary with airline tickets: finding a Eular path
'''

def findItinerary(tickets):
    # Initialize adjecency list
    from collections import defaultdict
    _tbl = defaultdict(list)
    for src, dst in tickets: 
        _tbl[src] += [dst]        
    _tbl = dict([(src, sorted(dsts)) for src, dsts in _tbl.items()])

    # The DFS will exhaust all the cycles first
    def dfs(curr):                
        itin = []
        while True:
            try: _dsts = _tbl[curr]
            except: _dsts = []
            # If there is not more out-going connections
            if [] == _dsts: break
            # Update the "global" connection 
            _tbl[curr] = _dsts[1:]
            itin = dfs(_dsts[0]) + itin

        return [curr] + itin

    return dfs('JFK')

    

def findItineraryCyclePatch(tickets):
    # Initialize adjecency list
    from collections import defaultdict
    _tbl = defaultdict(list)
    for src, dst in tickets: 
        _tbl[src] += [dst]        
    _tbl = dict([(src, sorted(dsts)) for src, dsts in _tbl.items()])

    def find_itin(curr):
        itin = []
        while True:
            itin += [curr]
            try: _dsts = _tbl[curr]
            except: _dsts = []
            if [] == _dsts: 
                break
            _tbl[curr] = _dsts[1:]
            curr = _dsts[0]
    
        return itin

    itin = find_itin('JFK')

    if len(tickets) == len(itin):
        return itin

    # There might be round trips
    i = 0
    while i < len(itin):
        src = itin[-i-1]
        if src not in _tbl or [] == _tbl[src]:
            i += 1; continue

        curr_itin = find_itin(src)
        itin = itin[:(-i-1)] + curr_itin + itin[-i:]
        i += 1
        
    return itin


def TEST(tickets, tgt):
    print('-----------')
    res = findItinerary(tickets)
    if res != tgt:
        print("Error")
        print(res)
        print("but expect")
        print(tgt)
    else:
        print("Ok")


TEST([["JFK","AAA"],["AAA","JFK"],["JFK","BBB"],["JFK","CCC"],["CCC","JFK"]],
     ['JFK', 'AAA', 'JFK', 'CCC', 'JFK', 'BBB'])
TEST([["EZE","TIA"],["EZE","HBA"],["AXA","TIA"],["JFK","AXA"],["ANU","JFK"],["ADL","ANU"],["TIA","AUA"],["ANU","AUA"],["ADL","EZE"],["ADL","EZE"],["EZE","ADL"],["AXA","EZE"],["AUA","AXA"],["JFK","AXA"],["AXA","AUA"],["AUA","ADL"],["ANU","EZE"],["TIA","ADL"],["EZE","ANU"],["AUA","ANU"]],
     ["JFK","AXA","AUA","ADL","ANU","AUA","ANU","EZE","ADL","EZE","ANU","JFK","AXA","EZE","TIA","AUA","AXA","TIA","ADL","EZE","HBA"])

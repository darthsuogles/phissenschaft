''' Data Stream Disjoint Intervals
'''

class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e

    def __str__(self):
        return '({}, {})'.format(self.start, self.end)


class SummaryRanges(object):

    def __init__(self):
        self.interval_init = []
        self.interval_fini = []


    def addNum(self, val):        
        from bisect import bisect_left
        if [] == self.interval_fini:
            self.interval_init = [val]
            self.interval_fini = [val]
            return

        n = len(self.interval_fini)
        i = bisect_left(self.interval_init, val)
            
        _inits = self.interval_init
        _finis = self.interval_fini
        if i < n and _inits[i] == val:
            return
        if i > 0 and val <= _finis[i-1]:
            return

        is_updated = False
        if i < n and val + 1 == _inits[i]:
            _inits[i] = val
            self.interval_init = _inits
            is_updated = True

        if i > 0 and _finis[i-1] + 1 == val:
            if i < n and val == _inits[i]:
                self.interval_init = _inits[:i] + _inits[(i+1):]
                _finis[i-1] = _finis[i]
                self.interval_fini = _finis[:i] + _finis[(i+1):]
            else:
                _finis[i-1] = val
                self.interval_fini = _finis
            is_updated = True

        if not is_updated:
            self.interval_init = _inits[:i] + [val] + _inits[i:]
            self.interval_fini = _finis[:i] + [val] + _finis[i:]
        
        
    def getIntervals(self):
        return [Interval(a, b) for a, b 
                in zip(self.interval_init, self.interval_fini)]


def TEST(nums):
    obj = SummaryRanges()
    for val in nums:    
        obj.addNum(val)
        print(list(map(str, obj.getIntervals())))
    print('-------------------------------------')

TEST([1, 3, 7, 2, 6, 93, 91, 90, 92])
TEST([1, 3, 7, 2, 6, 9, 4, 10, 5])

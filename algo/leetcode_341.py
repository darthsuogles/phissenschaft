''' Flatten nested list iterator
''' 

class NestedInteger(object):
    
    def __init__(self, nlst):
        self._list_or_int = nlst

    def isInteger(self):
        """
        @return True if this NestedInteger holds a single integer, rather than a nested list.
        :rtype bool
        """
        if int == type(self._list_or_int):
            return True
        return False


    def getInteger(self):
        """
        @return the single integer that this NestedInteger holds, if it holds a single integer
        Return None if this NestedInteger holds a nested list
        :rtype int
        """
        if self.isInteger():
            return self._list_or_int
        return None

   
    def getList(self):
        """
        @return the nested list that this NestedInteger holds, if it holds a nested list
        Return None if this NestedInteger holds a single integer
        :rtype List[NestedInteger]
        """
        if self.isInteger(): return None
        return self._list_or_int


class NestedIterator(object):

    def __init__(self, nestedList):
        """
        Initialize your data structure here.
        :type nestedList: List[NestedInteger]
        """
        self.curr = None
        self._list = nestedList

    def next(self):
        """
        :rtype: int
        """
        if self.hasNext():
            res = self.curr[0]
            self.curr = self.curr[1:]
            return res
            

    def _refill(self):
        if self.curr or not self._list: return

        def _flatten(_list_or_int):
            if _list_or_int is None or [] == _list_or_int:
                return []        
            if int == type(_list_or_int):
                return [_list_or_int]
            if NestedInteger == type(_list_or_int):
                if _list_or_int.isInteger():
                    return [_list_or_int.getInteger()]
                _ls = _list_or_int.getList()
            else:
                _ls = _list_or_int

            if not _ls: return []
            return _flatten(_ls[0]) + _flatten(_ls[1:])

        nlst = self._list
        curr = None
        while (not curr) and nlst:
            curr = _flatten(nlst[0])
            nlst = nlst[1:]
        self.curr = curr
        self._list = nlst


    def hasNext(self):
        """
        :rtype: bool
        """
        self._refill()
        if not self.curr: return False
        return True


def TEST(nlst):
    print('---------------------------')
    nestedList = [NestedInteger(e) for e in nlst]
    i, v = NestedIterator(nestedList), []
    while i.hasNext(): v.append(i.next())
    print(v)


TEST([[], 0, [], 0, 0, [0], [1,2], [1], 2, [2,3], []])
TEST([[1,1], 2, [1,1]])

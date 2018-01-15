
class RandomizedCollection(object):
    import random

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self._tbl = {} 
        self._lst = [-1]  # initialize to a certain size
        self._next_pos = 0

    def insert(self, val):
        """
        Inserts a value to the collection. 
        Returns true if the collection did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        n_pos = self._next_pos
        try: val_lst = self._tbl[val] + [n_pos]; res = False
        except: val_lst = [n_pos]; res = True
        if len(self._lst) == n_pos:
            self._lst.append(val)
        else:
            self._lst[n_pos] = val
        self._next_pos = n_pos + 1
        self._tbl[val] = val_lst
        return res

    def remove(self, val):
        """
        Removes a value from the collection. Returns true if the collection contained the specified element.
        :type val: int
        :rtype: bool
        """
        try: val_lst = self._tbl[val]
        except: return False
            
        s_pos = val_lst[-1]
        if 1 == len(val_lst): 
            del self._tbl[val]
        else:
            self._tbl[val] = val_lst[:-1]

        # Maintain the record
        last_pos = self._next_pos - 1            
        if last_pos != s_pos:
            swap_val = self._lst[last_pos]
            swap_val_lst = self._tbl[swap_val]
            swap_val_lst[-1] = s_pos
            self._tbl[swap_val] = sorted(swap_val_lst)  # cheating
            self._lst[s_pos] = swap_val

        self._next_pos = last_pos
        print("REMOVE: ", val, self._lst[:last_pos], self._tbl)
        return True


    def getRandom(self):
        """
        Get a random element from the collection.
        :rtype: int
        """        
        rand_pos = random.randint(0, self._next_pos - 1)
        return self._lst[rand_pos]


# Your RandomizedCollection object will be instantiated and called as such:
obj = RandomizedCollection()
print("null")
print(obj.insert(10))
print(obj.insert(10))
print(obj.insert(20))
print(obj.insert(20))
print(obj.insert(30))
print(obj.insert(30))
print(obj.remove(10))
print(obj.remove(10))
print(obj.remove(30))
print(obj.remove(30))

for i in range(7):
    print(obj.getRandom())

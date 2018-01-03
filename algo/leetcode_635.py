class LogSystem(object):
    """
    TODO: multiple level of range queries
    """

    def __init__(self):
        self.logs = []
        self.cmp_fn = None
        self.schema_idx = dict([(s, i) for i, s in 
                                enumerate(['Year', 'Month', 
                                           'Day', 'Hour', 'Minute', 'Second'])])
        self.min_id = -1
        self.max_id = -1

    def _parse(self, s):
        return tuple(int(el) for el in s.split(':'))

    def put(self, _id, timestamp):
        """
        :type id: int
        :type timestamp: str
        :rtype: void
        """
        #year, month, day, hh, mm, ss = self._parse(timestamp)
        #logs = self.log_shard[year][month]
        #logs.append(((day, hh, mm, ss), _id))
        self.logs.append((self._parse(timestamp), _id))
        self.min_id = min(self.min_id, _id)
        self.max_id = max(self.max_id, _id)

    def retrieve(self, s, e, gra):
        """
        :type s: str
        :type e: str
        :type gra: str
        :rtype: List[int]
        """
        from bisect import bisect_left, bisect_right

        cmp_idx = self.schema_idx[gra] + 1
        keys_init = self._parse(s)[:cmp_idx]
        keys_fini = self._parse(e)[:cmp_idx]
        logs = sorted([(stmp[:cmp_idx], _id) for stmp, _id in self.logs])
        p0 = bisect_left(logs, (keys_init, self.min_id))
        p1 = bisect_right(logs, (keys_fini, self.max_id))
        return [_id for _, _id in logs[p0:p1]]


obj = LogSystem()
obj.put(1, "2017:01:01:23:59:59");
obj.put(2, "2017:01:01:22:59:59");
obj.put(3, "2016:01:01:00:00:00");
print(obj.retrieve("2016:01:01:01:01:01","2017:01:01:23:00:00","Year"))
print(obj.retrieve("2016:01:01:01:01:01","2017:01:01:23:00:00","Hour"))

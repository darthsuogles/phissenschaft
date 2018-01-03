class Solution(object):
    def largestNumberAux(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        if len(nums) == 0: 
            return ""
        max_int_str = str(nums[0])
        max_int_idx = 0
        for idx, val in enumerate(nums[1:]):
            curr_str = str(val)
            v1 = int(curr_str + max_int_str)
            v2 = int(max_int_str + curr_str)
            if v1 > v2:
                max_int_str = curr_str
                max_int_idx = 1 + idx

        return max_int_str + self.largestNumberAux(
            nums[:max_int_idx] + nums[(1 + max_int_idx):])

    def largestNumberSort(self, nums):
        import functools
        def cmp_int_str(v1, v2):
            sv1 = str(v1); sv2 = str(v2)
            lhs = sv1 + sv2; rhs = sv2 + sv1
            if lhs == rhs:
                return 0
            if lhs < rhs:
                return 1
            if lhs > rhs:
                return -1    

        nums = sorted(nums, key=functools.cmp_to_key(cmp_int_str))
        return ''.join(map(str, nums))
    
    def largestNumber(self, nums):
        #max_int_str = self.largestNumberAux(nums)
        max_int_str = self.largestNumberSort(nums)
        if '0' == max_int_str[0]:
            return '0'
        return max_int_str
        
sol = Solution()
print(sol.largestNumber([1,2,3,4]))
print(sol.largestNumber([3, 30, 34, 5, 9]))
print(sol.largestNumber([0, 0]))
print(sol.largestNumber([0, 1, 0]))

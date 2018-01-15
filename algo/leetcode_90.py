''' Find subsets with duplicates
'''

def subsetsWithDup(nums):
    if not nums: return [[]]
    from collections import Counter
    
    def find_subsets(cntr):
        if not cntr: return [[]]

        a, cnt = cntr[0]
        res_sans = find_subsets(cntr[1:])
        return res_sans + [
            ([a] * i) + curr 
            for curr in res_sans
            for i in range(1, cnt+1)
        ]

    cntr = list(Counter(nums).items())
    return find_subsets(cntr)


print(subsetsWithDup([1,2,2]))

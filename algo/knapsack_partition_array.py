''' The partition problem 
    
    https://en.wikipedia.org/wiki/Partition_problem
'''

def get_partitions(nums):
    ''' Get the probability of seeing an equal partition
    '''
    if not nums: return 0
    n = len(nums)
    tbl_sack = []
    tot_sum = 0
    for a in nums:
        tbl_sack.append({})
        tot_sum += a
    if 0 != (tot_sum % 2):
        return 0
    W = tot_sum // 2
    
    def fill_sacks(i, w):
        ''' Fill the sack with exact values
            Return all possible combinations
        '''
        assert(0 <= i and i < n)
        if w < 0: 
            return []
        a = nums[i]
        if 0 == i:
            if w != a:  # need exact match of value
                return []
            return [[i]]

        try: return tbl_sack[i][w]
        except: pass

        res_sans = fill_sacks(i-1, w)
        res_with = [r + [i] for r in fill_sacks(i-1, w-a)]
        curr = res_sans + res_with
        tbl_sack[i][w] = curr
        return curr
    
    sack_combs = fill_sacks(n-1, W)
    print(list(('i = {}'.format(i), nums[i], 
                'j = {}'.format(j), nums[j]) 
               for i, j in sack_combs))

    # Validate result
    for sack in sack_combs:
        curr_tot = 0
        for i in sack:
            curr_tot += nums[i]
        assert(curr_tot == W)

    # There are 2 ** n ways in total 
    # Each sack combination can be swapped between
    # the two players
    return 2 * len(sack_combs) / (1 << (n - 1))


def TEST(nums):
    print(get_partitions(nums))

TEST([1,2,3,4])
TEST([1,2,2,1])


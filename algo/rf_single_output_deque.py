''' Single output deque
'''

def get_output_seq(nums):
    if not nums: return
    n = len(nums)
    deque = []

    ''' Auxiliary data '''
    # inds[a] => a's position in nums
    idx = [-1] + [e[0] for e in 
                  sorted(enumerate(nums), key=lambda e: e[1])]
    # stores the operations
    ops = []
    # stores the output sequence to check error
    result_seq = []

    i = 0
    for a in range(1, n + 1):
        if not deque:
            deque.append(a)
            ops.append('pushBack')            
        elif idx[a] < idx[deque[-1]]:
            deque.append(a)
            ops.append('pushBack')
        elif idx[a] > idx[deque[0]]:
            deque.insert(0, a)
            ops.append('pushFront')
        else:
            print('impossible')
            return

        while deque:
            if deque[-1] != nums[i]: 
                break
            ops.append('popBack')
            result_seq.append(deque.pop())            
            i += 1
            
    while deque:        
        ops.append('popBack')
        result_seq.append(deque.pop())

    assert result_seq == nums, 'result sequence not matching input'
    print(','.join(ops))


''' The main program '''
# Read one line from <STDIN>
from sys import stdin
line = stdin.readline()
nums = list(map(int, line.split(',')))
get_output_seq(nums)


    # def TEST(nums):
    #     print('----------------')
    #     get_output_seq(nums)


    # TEST([3,2,1])
    # TEST([1,2,3])
    # TEST([4,1,5,2,3])
    # TEST([5,1,4,2,3])

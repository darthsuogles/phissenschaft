''' Find the smallest unallocated natural number (>= 1)
'''

def next_server_number(server_numbers: list) -> int :
    if not server_numbers: return 1        
    sorted_numbers = sorted(set(server_numbers))
    for i, server_number in enumerate(sorted_numbers, 1):
        if i < server_number:
            return i
    return len(sorted_numbers) + 1

class Tracker(object):
    def __init__(self):
        # given a server name, returns a list allocated server with suffix
        from collections import defaultdict
        self.server_numbers = defaultdict(list)
        
    def allocate(self, server_base_name: str) -> str:
        ''' Allocated the next available server 
        '''
        server_numbers = self.server_numbers[server_base_name]
        next_number = next_server_number(server_numbers)
        self.server_numbers[server_base_name] += [next_number]
        return server_base_name + str(next_number)

    def deallocate(self, server_full_name: str):
        ''' Deallocate the given number
        ''' 
        nums = [ch for ch in server_full_name if '1' <= ch and ch <= '9']
        curr_number = int(''.join(nums))
        server_base_name = server_full_name[:-len(nums)]
        assert len(server_base_name) + len(nums) == len(server_full_name), \
            'server names must be properly parsed'
        if server_base_name not in self.server_numbers:
            # server name not exist, return
            print('Cannot find server base name', server_base_name)
            return
        
        server_numbers = self.server_numbers[server_base_name]
        server_numbers = [i for i in server_numbers if i != curr_number]        
        self.server_numbers[server_base_name] = server_numbers
        

def TEST(server_numbers, expected): 
    result = next_server_number(server_numbers)
    if result != expected:
        print('Error', result, 'but expect', expected)
    else:
        print('Ok')


print('TEST CASES')
TEST([5,3,1], 2)
TEST([5,4,1,2], 3)
TEST([3,2,1], 4)
TEST([2,3], 1)
TEST([], 1)
TEST([3,3,2], 1)
TEST([3,1,1], 2)

tracker = Tracker()
assert 'apibox1' == tracker.allocate('apibox')
assert 'apibox2' == tracker.allocate('apibox')
tracker.deallocate('apibox1')
tracker.deallocate('apibox2')
tracker.deallocate('apibox3')
#print(tracker.allocate('apibox'))
assert 'apibox1' == tracker.allocate('apibox')
assert 'sitebox1' == tracker.allocate('sitebox')

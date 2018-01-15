''' 
    n => 3 * n + 1  # if n is odd
    n => n // 2     # if n is even
'''

def longest_digit_seq(n):
    if n <= 0: return 0
    
    max_len = 0
    len_tbl = {}
    for i in range(2, n):
        curr_len = 0
        j = i
        while 1 != i:
            if i in len_tbl:
                curr_len += len_tbl[i]
                break
            curr_len += 1
            if 0 == i % 2:
                i = i // 2
            else:
                i = 3 * i + 1
            
        len_tbl[j] = curr_len
        max_len = max(max_len, curr_len)
        
    return max_len



print(longest_digit_seq(1000))

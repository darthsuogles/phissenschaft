''' Parallel postfix sum (scan)
'''

def post_scan_par(nums):
    if not nums: return None

    n = len(nums)  # assuming power-of-two length
    if 0 != (n & (n-1)): return None
    d = 1
    while d < n:
        d1 = d << 1
        # Each individual task can be done in parallel
        for i in range(0, n, d1):
            nums[i] += nums[i+d]
        d = d1

    print(nums)

    while d > 0:
        d1 = d << 1
        # Only the newly introduced terms need update
        for i in range(d, n, d1):
            if i + d >= n: break
            nums[i] += nums[i + d]            
        # for i in range(0, n, d1):
        #     if i < d: continue
        #     nums[i] += nums[i - d]   
        d >>= 1

    return nums


nums = [1,2,3,4,5,6,7,8]
tgt_psum = [0] * len(nums)
csum = 0
for i in range(len(nums)-1, -1, -1):
    csum += nums[i]
    tgt_psum[i] = csum

res_psum = post_scan_par(nums)
print(res_psum)
assert tgt_psum == res_psum

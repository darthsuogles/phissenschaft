
def wiggleSort(nums):
    if nums is None: return
    n = len(nums)
    if n <= 1: return 
    
    arr = nums
    nums = sorted(nums, reverse=True)
        
    k = n // 2
    prev = nums[k:]
    post = nums[:k]
    print(prev, post)
    buf = []
    i = 0; j = 0
    m = min(len(prev), len(post))
    while i < m:
        buf += [prev[i], post[i]]; i += 1
    while i < len(prev):
        buf += [prev[i]]; i += 1
    while i < len(post):
        buf += [post[i]]; i += 1

    i = 0
    while i < n:
        arr[i] = buf[i]; i += 1
    
        
def TEST(nums):
    wiggleSort(nums)
    is_less = True
    i = 1
    while i < len(nums):
        a = nums[i-1]; b = nums[i]
        if a == b: break
        if is_less and a > b: break
        if not is_less and a < b: break
        is_less = not is_less
        i += 1
    if i != len(nums):
        print("ERROR", nums)
    else:
        print("OK")
    

TEST([1, 5, 1, 1, 6])
TEST([1, 3, 2, 2, 3, 1])
TEST([1, 5, 1, 1, 6, 4])
TEST([1, 3, 2, 2, 3, 1])
TEST([4, 5, 5, 6])
TEST([1,2,3])
TEST([2,3,3,2,2,2,1,1])

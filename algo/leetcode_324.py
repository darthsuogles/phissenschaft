
def wiggleSortOutPlace(nums):
    """ Sort and reorg
    """
    if not nums: return
    arr = nums  # kep ref to the memory
    n = len(nums)
    res = [None] * n
    # The median elements must be separated as much as possible,
    # e.g. [6,5,5,4] => [5,6,4,5]
    #         #   #      #   #
    nums = sorted(nums, reverse=True)
    idx = n // 2
    res[::2], res[1::2] = nums[idx:], nums[:idx]
    for i in range(n): arr[i] = res[i]
    del res


def wiggleSort(nums):
    n = len(nums)
    if n <= 2: return
    from find_kth_smallest import find_kth

    mid = find_kth(nums, n // 2 + (n % 2))

    # Adding a phantom element to the end of
    # an even-sized sequence, so that the next
    # physical index will always land at the first
    # element of the array.
    # It ensures that accessing consecutive elements
    # will give us a run of the full sequence.
    def index(i):
        return (1 + 2 * i) % (n | 1)

    def swap(i, j):
        ix = index(i); jx = index(j);
        nums[ix], nums[jx] = nums[jx], nums[ix]

    # Standard three-way partition
    # The only difference is the way we treat indices
    i = 0; j = 0; k = n - 1
    while j <= k:
        p = nums[index(j)]
        if p > mid:
            swap(i, j); i += 1; j += 1
        elif p < mid:
            swap(j, k); k -= 1
        else:
            j += 1


def wiggleSortRef(nums):
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

    for i in range(n):
        arr[i] = buf[i]


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

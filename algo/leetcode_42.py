''' Trapping rain water
'''

def trap(height):
    trap_tot = 0
    idx_stack = []
    for i, h in enumerate(height):
        while idx_stack:
            top = height[idx_stack[-1]]
            if h < top:
                break
            idx_stack.pop()
            if idx_stack:
                j = idx_stack[-1]
                diff_height = min(h, height[j]) - top
                trap_tot += (i - j - 1) * diff_height

        idx_stack.append(i)

    return trap_tot


def trapDP(height):
    if not height: return 0
    n = len(height)
    if n <= 2: return 0

    # Build higher right side bar
    bar_height = [None] * n
    stq = []
    for i in range(n-1, -1, -1):
        h = height[i]
        while stq:
            j = stq[-1]
            if h <= height[j]:
                bar_height[i] = height[stq[0]]
                break
            stq.pop()
        else:
            stq.append(i)

    # Build higher left side bar
    trap_tot = 0
    stq = []
    for i in range(n):
        h = height[i]
        while stq:
            j = stq[-1]
            if h <= height[j]:
                prev_height = bar_height[i]
                if prev_height is not None:
                    curr_height = min(prev_height, height[stq[0]])
                    trap_tot += curr_height - h
                break
            stq.pop()
        else:
            stq.append(i)

    return trap_tot


def TEST(arr):
    ht = max(arr)
    for i in range(ht):
        line = []
        for h in arr:
            line.append('|' if i + h >= ht else ' ')
        print(' '.join(line))
    print('------------------------ max trap {}'.format(trap(arr)))


print('------TEST CASES---------')
TEST([0,1,0,2,1,0,1,3,2,1,2,1])
TEST([4,2,3])
TEST([4,9,4,5,3,2])
TEST([0,7,1,4,6])

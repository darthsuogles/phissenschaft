''' Largest rectangle in histogram
'''

def largestRectangleAreaDual(heights):
    if not heights: return 0
    n = len(heights)
    if 1 == n: return heights[0]

    def find_min_range(arr):
        stq = [0]
        min_bnd_rhs = [0] * n
        for i in range(1, n):
            h = arr[i]
            while stq:
                j = stq[-1]
                if h >= arr[j]: break
                min_bnd_rhs[j] = i - 1
                stq.pop(-1)

            stq.append(i)

        for j in stq:
            min_bnd_rhs[j] = n - 1
        return min_bnd_rhs


    min_bnd_rhs = find_min_range(heights)
    heights_rev = heights[::-1]
    min_bnd_lhs = [n-i-1 for i in find_min_range(heights_rev)[::-1]]
    # print(min_bnd_lhs)
    # print(heights)
    d_max = -1
    for k, (i, j) in enumerate(zip(min_bnd_lhs, min_bnd_rhs)):
        d = (j-i+1) * heights[k]
        d_max = max(d_max, d)
        #print(i, j, d, '[{} <-- {} --> {}]'.format(heights[i], heights[k], heights[j]))

    return d_max


def largestRectangleArea(heights):
    if not heights: return 0
    heights += [0]  # append 0, so that everything on stack will flush out
    stack = [-1]  # keep in ascending order
    max_area = 0
    # We keep a stack of ascending heights.
    # When a new height is encountered, we try to go back and eliminate
    # all the preceeding heights that are taller. All rectangles need a
    # minimum height. When a current maximum height is being eliminated,
    # we know that the largest rectangle where it is the shortest height
    # can be calculated via the current height and the one before it.
    for i, h in enumerate(heights):
        while stack:
            j = stack[-1]
            if heights[j] <= h:
                break  # no neg heights, smallest >= 0
            # Update the maximum area whenever we pop element
            stack.pop(); j0 = stack[-1]
            max_area = max(max_area, (i - j0 - 1) * heights[j])

        stack.append(i)

    return max_area


def TEST(heights):
    print(largestRectangleArea(heights))


TEST([2,1,1,1,5,5,6,2,3])
TEST([1,1])
TEST([2,2,4])

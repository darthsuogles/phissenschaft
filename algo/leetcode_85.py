''' Maximal rectangle (of area)

    1 0 1 0 0
    1 0 1 1 1
    1 1 1 1 1
    1 0 0 1 0

    Return 6
'''

def maximalRectangle(matrix):
    if not matrix: return 0
    m = len(matrix)
    n = len(matrix[0])
    if 0 == n: return 0

    def maximalArea(heights):
        stack = []  # keep in ascending order
        max_area = 0
        max_range = None
        for i, h in enumerate(heights):
            while stack:
                j = stack[-1]
                if heights[j] < h: break
                # Update the maximum area whenever we pop element
                stack = stack[:-1]
                j0 = -1 if not stack else stack[-1]
                curr = (i - j0 - 1) * heights[j]
                if curr > max_area:
                    max_area = curr
                    max_range = [j0 + 1, i - 1, heights[j]]

            stack.append(i)

        stack = [-1] + stack
        for j0, j in zip(stack[:-1], stack[1:]):
            curr = (n - j0 - 1) * heights[j]
            if curr > max_area:
                max_area = curr
                max_range = [j0 + 1, n - 1, heights[j]]

        return max_area, max_range

    # Can also do a top down scan and maintain the heights
    max_area = 0
    max_range = None
    heights = [0] * n
    for i, row in enumerate(matrix):  
        for j, elem in enumerate(row):
            if '1' == elem:
                heights[j] += 1
            else:
                heights[j] = 0

        curr_area, curr_range = maximalArea(heights)
        if curr_area > max_area:            
            max_area = curr_area
            max_range = [i] + curr_range

    return max_area, max_range
            

def TEST(matrix, tgt):
    res, rng = maximalRectangle(matrix)    
    if res != tgt:
        print('Error', res, 'but expect', tgt)
        return

    i1, j0, j1, h = rng
    i0 = i1 - h + 1
    for i, row in enumerate(matrix):
        row = list(row)
        if i0 <= i and i <= i1:
            for j in range(j0, j1 + 1):
                row[j] = '*'                            
        print(''.join(row))

    print('Ok', rng)


TEST([
    "10100",
    "10111",
    "11111",
    "10010"
], 6)

TEST(['10', '10'], 2)

TEST([
    "11111111",
    "11111110",
    "11111110",
    "11111000",
    "01111000"
], 21)

TEST([
    "0110010101",
    "0010101010",
    "1000010110",
    "0111111010",
    "0011111110",
    "1101011110",
    "0001100010",
    "1101100111",
    "0101101011"
], 10)

TEST([
    '11', 
    '11', 
    '01'
], 4)

TEST([
    "10100",
    "10111",
    "11111",
    "10010"
], 6)

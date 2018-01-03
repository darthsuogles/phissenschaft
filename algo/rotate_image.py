''' Rotate an image by 90 degree clockwise
'''

def rotateImage(a):
    if not a: return a
    n = len(a)
    # First diagonal mapping then flip row-wise
    for i in range(n):
        for j in range(i+1, n):
            a[j][i], a[i][j] = a[i][j], a[j][i]

    for i in range(n):
        row = a[i]
        for j in range(n // 2):
            row[j], row[n-j-1] = row[n-j-1], row[j]

    return a

def TEST(a):
    def print_matrix(a):
        print('-----------------')
        for row in a:
            print(' '.join(['{: 3d}'.format(i) for i in row]))

    print_matrix(a)
    print_matrix(rotateImage(a))


TEST([[1, 2, 3],
      [4, 5, 6],
      [7, 8, 9]])

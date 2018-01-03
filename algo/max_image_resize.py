
from math import ceil, floor

def max_image_resize(x, y, max_x, max_y):
    """
    Image size: m x n
    Window size: w x h
    """
    if 0 == max_x or 0 == max_y:
        return 0, 0

    def image_resize(ratio):
        rs_x = x * ratio
        rs_y = y * ratio
        if rs_x < 1.0 or rs_y < 1.0:
            return 0, 0
        ix = ceil(rs_x); iy = ceil(rs_y)
        if ix > max_x or iy > max_y:
            ix = floor(rs_x)
            iy = floor(rs_y)
        return ix, iy
    
    if x <= max_x:
        if y <= max_y:
            return x, y

        return image_resize(max_y / y)

    if y <= max_y:
        return image_resize(max_x / x)

    return image_resize(min(max_x / x, max_y / y))
        

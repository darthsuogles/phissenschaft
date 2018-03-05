"""
Strange shape
"""

import numpy as np
import matplotlib.pyplot as plt

R = 4
size = 100000

xs = np.random.random(size) * R
ys = np.random.random(size) * R

# plt.scatter(x, y); plt.show()
R2 = R * R
down_center = np.array([R / 2, R])
cnts = 0
for x, y in zip(xs, ys):
    if x * x + y * y > R2: continue
    _x = (x - down_center[0])
    _y = (y - down_center[1])
    if 4 * (_x * _x + _y * _y) > R2: continue
    cnts += 1

print(R2 * cnts / size)

"""
Bitwse operations
"""

for i in range(10000):
    assert ((i - 1) | i) & (i - 1) == (i - 1)

import bisect as bs
import numpy as np

s = np.array([2, 3, 1, 3, 4, 5])
sort_index = np.argsort(s)
print(sort_index)

i = bs.bisect_right(s[sort_index], 3)
print(i)
print(s[sort_index][max(0, i)])
print(s[sort_index][min(len(s) - 1, i + 1)])


import numpy as np

a = [1, 2, 3, 5, 8, 10, 15, 20, 30]
x = range(len(a))

gr = np.gradient(a, x, axis=None, edge_order=1)
print(gr)


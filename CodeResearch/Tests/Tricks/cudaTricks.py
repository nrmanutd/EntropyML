import time

import numpy as np
import cupy as cp

delta = 0

total = 3000

for i in range(total):
    np_array1 = np.random.randint(size=(500), low=0, high=1000)
    t1 = time.time()
    array1 = cp.asarray(np_array1)  # Преобразуем в CuPy-массив
    sorted = cp.argsort(array1)
    delta += (time.time() - t1)

print('Average time: ', delta/total)
print('Average time: ', delta)
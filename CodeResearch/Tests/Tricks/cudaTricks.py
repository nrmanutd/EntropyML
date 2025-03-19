import time

import numpy as np
import cupy as cp

total = 3000
numberOfObjects = np.arange(500, 5000, 500)
print(numberOfObjects)
results = np.zeros(len(numberOfObjects))

for n in range(len(numberOfObjects)):
    for i in range(total):
        np_array1 = np.random.randint(size=(numberOfObjects[n]), low=0, high=1000)
        t1 = time.time()
        array1 = cp.asarray(np_array1)  # Преобразуем в CuPy-массив
        sorted = cp.argsort(array1)
        results[n] += (time.time() - t1)

print('Average time: ', results/total)
print('Average time: ', results)
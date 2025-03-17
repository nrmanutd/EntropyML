import time

import numba
import numpy as np
from numba import cuda

@cuda.jit
def increment_by_one(an_array, result):
    pos = numba.cuda.grid(1)
    if pos < an_array.size:
        result[pos] = an_array[pos] + 1

@cuda.jit
def update(an_array, second_array):
    pos = numba.cuda.grid(1)
    if pos < second_array.size:
        an_array[second_array[pos]] = True

total = 20
N = 100
threadsperblock = 32
blockspergrid = (N + (threadsperblock - 1)) // threadsperblock

a = cuda.to_device(np.full(N, False, dtype=bool))
b = cuda.to_device([0, 1, 2, 3])
t = 0
toDevice = 0
for i in range(total):
    print (i)
    t1 = time.time()
    a = cuda.to_device(np.full((N, 3000), False, dtype=bool))
    toDevice += (time.time() - t1)

    t1 = time.time()
    update[blockspergrid, threadsperblock](a, b)
    t += (time.time() - t1)
    c = cuda.device_array_like(a)

print(a.copy_to_host())

print (t/total)
print(toDevice/total)


increment_by_one[blockspergrid, threadsperblock](a, c)

print(c.copy_to_host())


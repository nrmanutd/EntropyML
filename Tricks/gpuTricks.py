import numba
import numpy as np
from numba import cuda

@cuda.jit
def increment_by_one(an_array):
    pos = numba.cuda.grid(1)
    if pos < an_array.size:
        an_array[pos] += 1

an_array = np.ones(1000000000)
print(an_array)

threadsperblock = 32
blockspergrid = (an_array.size + (threadsperblock - 1)) // threadsperblock
increment_by_one[blockspergrid, threadsperblock](an_array)

print(an_array)



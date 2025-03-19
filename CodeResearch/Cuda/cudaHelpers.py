import cupy as cp
import numba as nb
import numpy as np
from numba import jit, cuda, prange


# Устройственная функция для преобразования float32 в uint32
@cuda.jit(device=True)
def float_to_int(f):
    u = np.uint32(np.float32(f).view(np.uint32))
    if f >= 0:
        return u | np.uint32(0x80000000)
    else:
        return ~u


# Кернел для создания ключей
@cuda.jit
def create_keys(matrix, weights, keys):
    i, j = cuda.grid(2)
    if i < matrix.shape[0] and j < matrix.shape[1]:
        val_int = float_to_int(matrix[i, j])
        weight_int = float_to_int(weights[i])
        keys[i, j] = (np.uint64(val_int) << 32) | np.uint64(weight_int)


def getSortedSetCuda(matrix, weights):
    N, d = matrix.shape

    # Переносим данные на GPU
    matrix_d = cp.asarray(matrix, dtype=cp.float32)
    weights_d = cp.asarray(-weights, dtype=cp.float32)
    keys_d = cp.zeros((N, d), dtype=cp.uint64)

    # Настраиваем сетку и блоки
    block_size = (32, 32)
    grid_size = ((N + block_size[0] - 1) // block_size[0],
                 (d + block_size[1] - 1) // block_size[1])

    # Запускаем кернел для создания ключей
    create_keys[grid_size, block_size](matrix_d, weights_d, keys_d)

    # Создаем результирующую матрицу
    result = cp.empty((N, d), dtype=cp.int64)

    # Сортируем каждый столбец
    for j in range(d):
        result[:, j] = cp.argsort(keys_d[:, j])

    return result.get()

@jit(nopython=True, parallel=True)
def updateSortedSetNumba(matrix, indexes):
    nFeatures = matrix.shape[1]
    nObjects = matrix.shape[0]
    newObjects = len(indexes)

    result = np.zeros((newObjects, nFeatures), dtype=np.int32)

    map = dict()
    for i in range(newObjects):
        map[indexes[i]] = i

    currentState = np.zeros(nFeatures, dtype=np.int32)

    for iFeature in prange(nFeatures):
        for iObject in range(nObjects):
            curObjectIdx = matrix[iObject, iFeature]
            if curObjectIdx in map:
                result[currentState[iFeature], iFeature] = map[curObjectIdx]
                currentState[iFeature] += 1

    return result
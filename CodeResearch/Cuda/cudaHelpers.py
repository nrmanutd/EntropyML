import cupy as cp
import numpy as np
from numba import jit, cuda, prange, njit

from CodeResearch.DiviserCalculation.diviserHelpers import iv2s


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

# Устройственная функция для сортировки слиянием
@cuda.jit(device=True)
def merge_sort(arr, weights, indices, start, end):
    if end - start > 1:
        mid = (start + end) // 2
        merge_sort(arr, weights, indices, start, mid)
        merge_sort(arr, weights, indices, mid, end)
        merge(arr, weights, indices, start, mid, end)

# Устройственная функция для слияния двух отсортированных частей
@cuda.jit(device=True)
def merge(arr, weights, indices, start, mid, end):
    temp = cuda.local.array(shape=1024, dtype=np.float32)  # Временный массив для значений
    temp_weights = cuda.local.array(shape=1024, dtype=np.float32)  # Временный массив для весов
    temp_indices = cuda.local.array(shape=1024, dtype=np.int32)  # Временный массив для индексов
    i = start
    j = mid
    k = 0

    # Слияние двух частей
    while i < mid and j < end:
        if arr[i] < arr[j] or (arr[i] == arr[j] and weights[i] < weights[j]):
            temp[k] = arr[i]
            temp_weights[k] = weights[i]
            temp_indices[k] = indices[i]
            i += 1
        else:
            temp[k] = arr[j]
            temp_weights[k] = weights[j]
            temp_indices[k] = indices[j]
            j += 1
        k += 1

    # Копируем оставшиеся элементы
    while i < mid:
        temp[k] = arr[i]
        temp_weights[k] = weights[i]
        temp_indices[k] = indices[i]
        i += 1
        k += 1
    while j < end:
        temp[k] = arr[j]
        temp_weights[k] = weights[j]
        temp_indices[k] = indices[j]
        j += 1
        k += 1

    # Копируем результат обратно в исходные массивы
    for idx in range(k):
        arr[start + idx] = temp[idx]
        weights[start + idx] = temp_weights[idx]
        indices[start + idx] = temp_indices[idx]

# CUDA-ядро для сортировки каждого столбца
@cuda.jit
def sort_columns_kernel(matrix, weights, sorted_indices):
    col = cuda.blockIdx.x  # Текущий столбец
    tid = cuda.threadIdx.x  # Текущий поток
    n = matrix.shape[0]  # Количество объектов

    # Копируем столбец и веса в локальную память
    local_arr = cuda.local.array(shape=1024, dtype=np.float32)
    local_weights = cuda.local.array(shape=1024, dtype=np.float32)
    local_indices = cuda.local.array(shape=1024, dtype=np.int32)

    # Обрабатываем часть столбца
    chunk_size = 1024  # Количество элементов, обрабатываемых одним блоком
    start = tid * chunk_size
    end = min(start + chunk_size, n)

    for i in range(start, end):
        local_arr[i - start] = matrix[i, col]
        local_weights[i - start] = weights[i]
        local_indices[i - start] = i  # Изначальные индексы

    # Сортируем часть столбца
    merge_sort(local_arr, local_weights, local_indices, 0, end - start)

    # Сохраняем отсортированные индексы в результирующий массив
    for i in range(start, end):
        sorted_indices[i, col] = local_indices[i - start]

def sort_matrix(matrix, weights):
    n, d = matrix.shape

    # Копируем матрицу и веса на устройство
    d_matrix = cuda.to_device(matrix)
    d_weights = cuda.to_device(weights)

    # Создаем массив для отсортированных индексов на устройстве
    d_sorted_indices = cuda.device_array((n, d), dtype=np.int32)

    # Задаем размер блока и сетки
    threads_per_block = 1024  # Количество потоков в блоке
    blocks_per_grid = d  # Количество блоков равно количеству столбцов

    # Запускаем ядро
    sort_columns_kernel[blocks_per_grid, threads_per_block](d_matrix, d_weights, d_sorted_indices)

    # Копируем результат обратно на хост
    sorted_indices = d_sorted_indices.copy_to_host()
    return sorted_indices


@jit(nopython=True)
def updateSortedSetByBucketNumba(matrix, indexes):
    nFeatures = matrix.shape[1]
    newObjects = len(indexes)

    result = np.zeros((newObjects, nFeatures), dtype=np.int32)

    for iFeature in prange(nFeatures):
        r = bucket_sort_with_order(matrix[:, iFeature], indexes)
        result[:, iFeature] = r

    return result

@njit
def bucket_sort_with_order(sorted_indices, subset_ids, num_buckets=100):
    N = len(sorted_indices)  # Общее количество элементов
    K = len(subset_ids)  # Количество элементов в подмножестве

    # Создаем словарь для быстрого доступа к позициям идентификаторов

    position_map = {idx: pos for pos, idx in enumerate(sorted_indices)}
    elements_map = {idx: pos for pos, idx in enumerate(subset_ids)}

    # Вычисляем максимальный размер каждого бакета
    max_bucket_size = (N + num_buckets - 1) // num_buckets

    # Создаем массив для хранения бакетов
    buckets = np.zeros((num_buckets, max_bucket_size), dtype=np.int32)

    # Создаем массив для хранения количества элементов в каждом бакете
    bucket_counts = np.zeros(num_buckets, dtype=np.int32)

    # Размещаем элементы по бакетам
    for idx in subset_ids:
        pos = position_map[idx]  # Позиция элемента в отсортированном массиве
        bucket_index = pos * num_buckets // N  # Вычисляем номер бакета
        buckets[bucket_index, bucket_counts[bucket_index]] = idx
        bucket_counts[bucket_index] += 1

    # Сортируем каждый бакет
    for i in range(num_buckets):
        if bucket_counts[i] > 0:
            buckets[i, :bucket_counts[i]] = np.sort(buckets[i, :bucket_counts[i]])

    # Конкатенируем бакеты
    result = np.zeros(K, dtype=np.int32)
    index = 0

    for i in range(num_buckets):
        if bucket_counts[i] > 0:
            result[index:index + bucket_counts[i]] = buckets[i, :bucket_counts[i]]
            index += bucket_counts[i]

    for i in range(len(result)):
        result[i] = elements_map[result[i]]

    return result

@jit(nopython=True, parallel=True)
def convertSortedSetToBackMap(sortedSet):
    nFeatures = sortedSet.shape[1]
    nObjects = sortedSet.shape[0]

    result = np.zeros((nObjects, nFeatures), dtype=np.int32)

    for iFeature in prange(nFeatures):
        for iObject in range(nObjects):
            result[sortedSet[iObject, iFeature], iFeature] = iObject

    return result

def updateSortedSetCupy(pseudoSortedSet):

    nFeatures = pseudoSortedSet.shape[1]
    nObjects = pseudoSortedSet.shape[0]

    result = np.zeros((nObjects, nFeatures), dtype=np.int32)

    for iFeature in range(nFeatures):
        arr = cp.asarray(pseudoSortedSet[:, iFeature])
        sorted = cp.argsort(arr)

        result[:, iFeature] = sorted.get()

    return result

@cuda.jit
def filterSortedSetByCuda(sortedSet, nObjects, nFeatures, index, result):
    thread_idx = cuda.grid(1)

    if thread_idx < nFeatures:
        currentIndex = 0
        for i in range(nObjects):
            curObjectIndex = index[sortedSet[i, thread_idx]]
            if curObjectIndex != -1:
                result[currentIndex, thread_idx] = curObjectIndex
                currentIndex += 1

    return

def filterSortedSetByIndex(sortedSet_device, nObjects, nFeatures, index):

    newObjects = len(index)

    targetIndex = np.full(nObjects, -1, dtype=np.int32)
    for i in range(len(index)):
        targetIndex[index[i]] = i

    index_device = cuda.to_device(targetIndex)
    result = np.zeros((newObjects, nFeatures), dtype=np.int32)
    result_device = cuda.to_device(result)

    threads_per_block = 64
    blocks_per_grid = (nFeatures + threads_per_block - 1) // threads_per_block

    filterSortedSetByCuda[blocks_per_grid, threads_per_block](sortedSet_device, nObjects, nFeatures, index_device, result_device)
    cuda.synchronize()

    result = result_device.copy_to_host()
    return result
import numpy as np
import cupy as cp

# Создаем два массива в NumPy
np_array1 = np.array([2, 2, 3, 4])
np_array2 = np.array([2, 2, 2, 5])

# Преобразуем массивы NumPy в CuPy-массивы
array1 = cp.asarray(np_array1)  # Преобразуем в CuPy-массив
array2 = cp.asarray(np_array2)  # Преобразуем в CuPy-массив

# Покомпонентное сравнение
comparison_result = array1 < array2  # Массив булевых значений

# Проверяем, есть ли хотя бы один True
flag = cp.any(comparison_result).item()  # Преобразуем в Python bool

print("Результат сравнения:", flag)
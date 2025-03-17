import numpy as np

from CodeResearch.Cuda.cudaHelpers import sort_columns

# Пример использования
matrix_np = np.array([
    [3.0, 1.0],
    [2.0, 4.0],
    [2.0, 3.0],
    [1.0, 2.0]], dtype=np.float32)
weights_np = np.array([0.5, 0.4, 0.3, 0.4], dtype=np.float32)
result = sort_columns(matrix_np, weights_np)
print(result.get())  # Переносим результат на CPU для вывода
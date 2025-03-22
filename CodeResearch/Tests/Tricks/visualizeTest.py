import numpy as np
import matplotlib.pyplot as plt

from CodeResearch.Visualization.VisualizeAndSaveCommonTopSubsamples import visualizeAndSaveKSForEachPair

# Пример данных: массив статистик для нескольких пар классов
# Предположим, у нас есть 5 пар классов, и для каждой пары 100 значений статистики
np.random.seed(42)
numClasses = 45
data = [np.random.normal(loc=i, scale=1.0, size=100) for i in range(numClasses)]
labels = [f"{x}/{y}" for x in range(10) for y in range(x + 1)]

visualizeAndSaveKSForEachPair(data, labels)
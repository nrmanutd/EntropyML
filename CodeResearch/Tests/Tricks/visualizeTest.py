import numpy as np
import matplotlib.pyplot as plt

from CodeResearch.Visualization.VisualizeAndSaveCommonTopSubsamples import visualizeAndSaveKSForEachPair, \
    visualizeAndSaveKSForEachPairAndTwoDistributions

# Пример данных: массив статистик для нескольких пар классов
# Предположим, у нас есть 5 пар классов, и для каждой пары 100 значений статистики
np.random.seed(42)
numClasses = 45
data1 = [np.random.normal(loc=i, scale=1.0, size=100) for i in range(numClasses)]
data2 = [np.random.normal(loc=i+2, scale=1.0, size=100) for i in range(numClasses)]

labels = [f"{x}/{y}" for x in range(10) for y in range(x + 1)]

#visualizeAndSaveKSForEachPair(data1, labels)
visualizeAndSaveKSForEachPairAndTwoDistributions(data1, data2, labels, 'task', '0_0')
import numpy as np

from CodeResearch.Visualization.VisualizeAndSaveCommonTopSubsamples import visualizeAndSaveKSForEachPair, \
    visualizeAndSaveKSForEachPairAndTwoDistributions
from CodeResearch.Visualization.saveDataForVisualization import deserialize_labeles_list_of_arrays

# Десериализация
ksData = deserialize_labeles_list_of_arrays("..\\PValuesFigures\\PValuesLogs\\KS_mnist_100_2_1.txt")
loaded_task=ksData[2]
loaded_attempts=ksData[3]
labels = ksData[1]
ks = ksData[0]

NNData = deserialize_labeles_list_of_arrays("..\\PValuesFigures\\PValuesLogs\\NN_mnist_10_2_1.txt")
NNloaded_task=NNData[2]
NNloaded_attempts=NNData[3]
NN = NNData[0]

print("\nЗагруженные данные:")
print(f"Название задачи: {loaded_task}")
print(f"Количество попыток: {loaded_attempts}")
print("Массивы:")
for arr, name in zip(ks, labels):
    print(f"{name}: {arr}")

visualizeAndSaveKSForEachPair(ks, labels, loaded_task, loaded_attempts, 'total', '..\\PValuesFigures\\')
visualizeAndSaveKSForEachPairAndTwoDistributions(ks, NN, labels, '{}_{} and {}_{}'.format(loaded_task, loaded_attempts, NNloaded_task, NNloaded_attempts), 'total', '..\\PValuesFigures\\')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from CodeResearch.Visualization.saveDataForVisualization import deserialize_labeles_list_of_arrays

data = deserialize_labeles_list_of_arrays('../../PValuesFigures/PValueLogs/KS_entropy_megamarket_200_21_6.txt')
dataf = deserialize_labeles_list_of_arrays('../../PValuesFigures/PValueLogs/KS_frequency_megamarket_200_21_6.txt')
datae = deserialize_labeles_list_of_arrays('../../PValuesFigures/PValueLogs/KS_error_megamarket_200_21_6.txt')
#entropies = np.array(data[0][0])
p = np.array(dataf[0][0])
entropies = np.minimum(p, 1 - p)
minBayes = np.mean(entropies)
e = datae[0][0][0]
print(f'Min bayes: {e}')
#frequences = np.array(dataf[0][0])

x = pd.read_parquet("../../Data/megamarket/sampled_10k.parquet")
y = np.array(x['cat_level_1'])

enc = LabelEncoder()
target = enc.fit_transform(np.ravel(y))

firstClass = 21
secondClass = 6

iObjects = list(np.where(target == firstClass)[0])
jObjects = list(np.where(target == secondClass)[0])
objectsIdx = iObjects + jObjects

# Построение гистограммы
plt.figure(figsize=(10, 6))

e1 = entropies[0:len(iObjects)]
e2 = entropies[len(iObjects):len(entropies)]
plt.hist(e1, bins=30, alpha=0.5, label=f'Распределение {firstClass} (соусы)', color='blue', density=True)
plt.hist(e2, bins=30, alpha=0.5, label=f'Распределение {secondClass} (кондитеры)', color='red', density=True)

# Настройки графика
plt.title('Гистограмма распределения данных', fontsize=14)
plt.xlabel('Значения', fontsize=12)
plt.ylabel('Частота', fontsize=12)
plt.grid(alpha=0.3)
plt.legend()

plt.show()

xx1 = np.array(list(zip(x['name'], x['cat_level_1'])))[iObjects]
xx2 = np.array(list(zip(x['name'], x['cat_level_1'])))[jObjects]

e1idx = np.argsort(-np.array(e1))
e2idx = np.argsort(-np.array(e2))
subSample1 = xx1[e1idx]
subSample2 = xx2[e2idx]
print(f'{firstClass} (соусы)')
print(subSample1[0:20])
print(e1[e1idx][0:20])
print('=======================')
print(subSample1[-20:])
print(e1[e1idx][-20:])
print(f'{secondClass} (кондитеры)')
print(subSample2[0:20])
print(e2[e2idx][0:20])
print('=======================')
print(subSample2[-20:])
print(e2[e2idx][-20:])
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from scipy import stats
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

ll = [185, 370, 555, 741, 926]
y = []

for l in ll:
    datae1 = deserialize_labeles_list_of_arrays('../../PValuesFigures/PValueLogs/KS_error_megamarket_1000_21_6_{0}.txt'.format(l))
    y.append(datae1[0][0][0])

x = 1/np.array([0.1, 0.2, 0.3, 0.4, 0.5])
y = np.array(y)

# Проверяем, что массивы одинаковой длины
if len(x) != len(y):
    raise ValueError("Массивы должны быть одинаковой длины")

# Берем логарифмы (добавляем маленькое значение чтобы избежать log(0))
log_x = np.log(x + 1e-10)
log_y = np.log(y + 1e-10)

# Линейная регрессия в логарифмических координатах
slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)

# Уравнение прямой: log(y) = slope * log(x) + intercept
# Что эквивалентно: y = exp(intercept) * x^slope

# Коэффициенты для исходных координат
A = np.exp(intercept)  # коэффициент перед x^slope
power = slope          # степень

# Создаем график
plt.figure(figsize=(12, 8))

# 1. Исходные данные
plt.subplot(2, 1, 1)
plt.scatter(x, y, color='blue', s=50, label='Исходные точки', alpha=0.7)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Исходные данные', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.yscale('log')  # логарифмическая шкала по y
plt.xscale('log')  # логарифмическая шкала по x

# 2. Логарифмические координаты с прямой
plt.subplot(2, 1, 2)
plt.scatter(log_x, log_y, color='red', s=50, label='Точки (log x, log y)', alpha=0.7)

# Строим линию регрессии
x_fit = np.linspace(min(log_x), max(log_x), 100)
y_fit = slope * x_fit + intercept
plt.plot(x_fit, y_fit, 'black', linewidth=2,
         label=f'Прямая: log(y) = {slope:.3f}·log(x) + {intercept:.3f}, k = {2*slope:.3f}')

plt.xlabel('log(x)', fontsize=12)
plt.ylabel('log(y)', fontsize=12)
plt.title('Логарифмические координаты с линейной регрессией', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

# Добавляем уравнение в виде аннотации
equation_text = (f'Уравнение в исходных координатах:\n'
                 f'y = {A:.3f}·x^{{{power:.3f}}}\n'
                 f'R² = {r_value**2:.4f}')

plt.figtext(0.5, 0.01, equation_text, ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()
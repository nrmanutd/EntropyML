import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from CodeResearch.StatisticsAnalyzer.dataLoader import loadNNsyncData, loadKSsyncData

taskName = 'mnist'

ksData = loadKSsyncData(taskName)
nnData = loadNNsyncData(taskName)
labels = ksData[1]

pairs = []

for i in range(len(ksData[0])):
    pairs.append((ksData[0][i], nnData[0][i]))

# Создание сетки графиков: 5 строк, 9 столбцов
fig, axes = plt.subplots(5, 9, figsize=(18, 10))
axes = axes.flatten()  # Преобразуем в одномерный массив для удобства

# Цикл по всем парам массивов
for i, (x, y) in enumerate(pairs):
    x = np.array(x)
    y = np.array(y)
    # Вычисление линейной регрессии
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r_squared = r_value ** 2

    # Построение графика рассеяния и линии регрессии
    axes[i].scatter(x, y, alpha=0.5)
    axes[i].plot(x, intercept + slope * x, 'r', label=f'R² = {r_squared:.2f}')

    # Выделение фона, если зависимость незначима (p-value >= 0.05)
    if p_value >= 0.05:
        axes[i].set_facecolor('lightcoral')  # Светло-красный фон

    # Настройка заголовка и легенды
    axes[i].set_title(f'Пара {labels[i]}')
    axes[i].legend()

# Общие настройки графика
plt.tight_layout()
plt.suptitle('Графики зависимости и R² для 45 пар массивов', y=1.02)
plt.show()
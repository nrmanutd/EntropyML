import numpy as np
import matplotlib.pyplot as plt

from CodeResearch.StatisticsAnalyzer.dataLoader import loadData
from CodeResearch.Visualization.metricCalcultation import calculateMetric

directory = "C:\\Current\\Work\\Science\\CodeResearch\\PValuesFigures\\PValueLogs_Serie_Blobs"
data = loadData(directory)

xSerie = range(1, 1 + len(data))
metrics = []

for k, d in data.items():
    metrics.append(calculateMetric(d['ksData'], d['pData'])[0])

metrics = np.array(metrics)

x_values = xSerie
y_values = metrics

# Создание графика
plt.figure(figsize=(8, 6))  # Размер графика
plt.plot(x_values, y_values, color='purple', linewidth=3, linestyle='-', label='Зависимость')  # Жирная фиолетовая линия
plt.scatter(x_values, y_values, color='purple', marker='o', s=100)  # Круглые точки

# Настройка осей
plt.xlabel('SD', fontsize=12)
plt.ylabel('Величина', fontsize=12)
plt.xticks(x_values)  # Отображаем только целые числа от 1 до 9
plt.ylim(0, 1)  # Ограничиваем ось Y от 0 до 1
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Показываем и очищаем
plt.show()
plt.close()


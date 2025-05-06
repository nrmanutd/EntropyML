import numpy as np
import matplotlib.pyplot as plt

from CodeResearch.StatisticsAnalyzer.dataLoader import loadData
from CodeResearch.Visualization.metricCalcultation import calculateMetric

def updateArray(arr):
    high = 18
    low = 428

    delta = low - high

    return (np.ones(9) * low - arr) / delta

def plotValues(plt, x, y, c, l, m='x'):
    plt.plot(x, y, color=c, linestyle='-', label=l, marker=m, markersize=8, markeredgewidth=2)
    return

directory = "C:\\Current\\Work\\Science\\CodeResearch\\PValuesFigures\\PValueLogs_Serie_Blobs"
data = loadData(directory)

xSerie = range(1, 1 + len(data))
metrics = []

for k, d in data.items():
    ksMedians = 1 - np.array([np.average(l) for l in d['ksData']])[0]
    metrics.append(ksMedians)
    #metrics.append(calculateMetric(d['ksData'], d['pData'])[0])

metrics = np.array(metrics)

n2 = updateArray([424, 403, 370, 335, 307, 286, 271, 261, 254])
n4 = updateArray([424, 421, 397, 360, 332, 303, 297, 277, 274])
t1 = updateArray([424, 410, 355, 295, 252, 224, 205, 188, 180])
lsc = updateArray([218, 87, 28, 21, 19, 18, 18, 18, 18])
density = updateArray([96, 90, 83, 76, 69, 66, 65, 62, 61])
dsi = updateArray([424, 317, 213, 142, 104, 82, 65, 58, 47])
rtd = updateArray([20, 37, 83, 144, 187, 224, 252, 263, 278])
rs = updateArray([373, 293, 227, 176, 139, 112, 93, 79, 71])

x_values = xSerie
y_values = metrics

# Создание графика

fig, ax = plt.subplots(figsize=(16, 12))
ax.plot(x_values, y_values, color='maroon', linewidth=3, markeredgewidth=3, linestyle='-', label='MKS (proposed)', marker='s', markerfacecolor='white', markeredgecolor='maroon', markersize=10)  # Жирная фиолетовая линия

plotValues(ax, x_values, n2, 'darkblue', 'N2')
#plotValues(ax, x_values, n4, '#B8860B', 'N4')
#plotValues(ax, x_values, t1, '#87CEEB', 'T1')
plotValues(ax, x_values, lsc, 'orange', 'LSC')
plotValues(ax, x_values, density, 'purple', 'Density')
plotValues(ax, x_values, dsi, 'green', 'DSI')
plotValues(ax, x_values, rs, 'red', 'RS')
#plotValues(ax, x_values, rtd, 'black', 'rTD', 'o')

# Настройка осей
ax.set_xlabel('Dataset # (Cluster SD)', fontsize=12)
ax.set_ylabel('Complexity measure', fontsize=12)
ax.set_xticks(x_values)  # Отображаем только целые числа от 1 до 9
ax.set_ylim(-0.02, 1.05)  # Ограничиваем ось Y от 0 до 1
ax.grid(True, linestyle='--', alpha=0.7)

# легенда под графиком
#ax.legend(loc='upper center',           # якорь — центр сверху
#          bbox_to_anchor=(0.5, -0.15),  # 0.5 по X, чуть ниже оси по Y
#          ncol=6,                       # два столбца
#          frameon=False)                # без рамки

ax.legend()
plt.tight_layout()

# Показываем и очищаем
plt.show()
plt.close()


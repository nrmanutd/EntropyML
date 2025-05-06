import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import datasets

from CodeResearch.StatisticsAnalyzer.dataLoader import loadData
from CodeResearch.Visualization.metricCalcultation import calculateMetric
from CodeResearch.dataSets import make_xor, make_spirals, make_random


# Функция для построения scatter plot
def plot_scatter(ax, X, y, title):
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Class 0', alpha=0.6, s=5)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='orange', label='Class 1', alpha=0.6, s=5)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title(f'{title} - Dataset')
    ax.legend()
    ax.grid(True)

# Функция для построения распределений
def plot_distributions(ax, dist1, dist2, title):
    sns.kdeplot(dist1, ax=ax, color='#0000FF', label='KS', fill=True, alpha=0.3)
    sns.kdeplot(dist2, ax=ax, color='#FFA500', label='Random', fill=True, alpha=0.3)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(f'{title} - Distributions')
    ax.legend()
    ax.grid(True)

def generateDataSet(taskName, sd = 1):
    nSamples = 2000

    if 'blobs' in taskName:
        return datasets.make_blobs(n_samples=nSamples, centers=2, n_features=2, random_state=42, cluster_std=sd)

    if 'moons' in taskName:
        return datasets.make_moons(n_samples=nSamples, noise=0.05, random_state=42)

    if 'circles' in taskName:
        return datasets.make_circles(n_samples=nSamples, factor=0.5, noise=0.05, random_state=42)

    if 'xor' in taskName:
        return make_xor(nSamples)

    if 'spirals' in taskName:
        return make_spirals(nSamples, noise=0.05)

    if 'random' in taskName:
        return make_random(nSamples)

    raise ValueError(f'Error in processing task {taskName}')

directory = "C:\\Current\\Work\\Science\\CodeResearch\\PValuesFigures\\PValueLogs_TestTasks_10000"
#directory = "C:\\Current\\Work\\Science\\CodeResearch\\PValuesFigures\\PValueLogs_Serie_Blobs"
#directory = "C:\\Current\\Work\\Science\\CodeResearch\\PValuesFigures\\PValueLogs_TargetTasks"
data = loadData(directory)

dataSets = []
distributions = []
metrics = []
taskNames = []

dataSetsOrder = ['blobs', 'moons', 'circles_sklearn', 'xor', 'spirals', 'random_sklearn']

for taskName in dataSetsOrder:
    d = data[taskName]
    dataSets.append(generateDataSet(taskName))
    distributions.append((d['ksData'], d['pData']))
    #if taskName == 'random_sklearn':
    #    metrics.append(0.9998)
    #else:
    ksMedians = np.array([np.average(k) for k in d['ksData']])
    metrics.append(1 - ksMedians[0])
        #metrics.append(calculateMetric(d['ksData'], d['pData'])[0])
    taskNames.append(taskName)

#for k, d in data.items():
#    dataSets.append(generateDataSet(k))
#    distributions.append((d['ksData'], d['pData']))
#    metrics.append(calculateMetric(d['ksData'], d['pData'])[0])
#    taskNames.append(k)

print(metrics)

# Создание фигуры 3x4
fig, axes = plt.subplots(3, 4, figsize=(16, 8))

# Итерация по датасетам
for idx, ((X, y), (dist1, dist2)) in enumerate(zip(dataSets, distributions)):
    # Вычисляем индексы для scatter и distribution подграфиков
    row = idx // 2  # Строка
    col = (idx % 2) * 2  # Столбцы для scatter (0 или 2)

    # Scatter plot
    plot_scatter(axes[row, col], X, y, f'{taskNames[idx]}')

    # Distribution plot (соседний столбец)
    plot_distributions(axes[row, col + 1], dist1, dist2, f'{taskNames[idx]}')

    ax_dist = axes[row, col + 1]
    ax_dist.text(0.5, 0.95, f'{max(0, metrics[idx]):.4f}',
                 transform=ax_dist.transAxes,
                 fontsize=12, fontweight='bold',
                 ha='center', va='top')

# Настройка компоновки
plt.tight_layout()
plt.suptitle('Datasets and distributions', y=1.02)
plt.savefig(f'Datasets_distribution_complexity.png', dpi=150, bbox_inches='tight')
plt.close(fig)
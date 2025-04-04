import numpy as np
from scipy.stats import kendalltau, spearmanr, mannwhitneyu, linregress

from CodeResearch.StatisticsAnalyzer.dataLoader import loadKSData, loadNNData, loadPermutationData, loadKSsyncData, \
    loadNNsyncData

#taskName = 'mnist'
taskName = 'cifar'

ksData = loadKSData(taskName)
nnData = loadNNData(taskName)
pData = loadPermutationData(taskName)
#permutationData = loadPermutationData(taskName)

labels = ksData[1]

upper = 0.9
lower = 1-upper

ksMedians = np.array([np.average(k) for k in ksData[0]])
permutationMedians = np.array([np.average(k) for k in pData[0]])

ksMedians = ksMedians - permutationMedians

nnMedians = np.array([np.average(k) for k in nnData[0]])
#ksMedians = np.array([np.quantile(k, upper) - np.quantile(k, lower) for k in ksData[0]])
#nnMedians = np.array([np.quantile(k, upper) - np.quantile(k, lower) for k in nnData[0]])

tau, p_value_kendall = kendalltau(ksMedians, nnMedians)
print(f"Корреляция Кендалла:")
print(f"Коэффициент Tau: {tau:.4f}")
print(f"p-value: {p_value_kendall:.4f}")
if p_value_kendall < 0.05:
    print("Связь значима (p < 0.05)")
else:
    print("Связь не значима (p >= 0.05)")

# Для сравнения: Корреляция Спирмена
corr, p_value_spearman = spearmanr(ksMedians, nnMedians)
print(f"\nКорреляция Спирмена:")
print(f"Коэффициент корреляции: {corr:.4f}")
print(f"p-value: {p_value_spearman:.4f}")
if p_value_spearman < 0.05:
    print("Связь значима (p < 0.05)")
else:
    print("Связь не значима (p >= 0.05)")

threshold = np.median(ksMedians)
low_sep_group = nnMedians[ksMedians <= threshold]
high_sep_group = nnMedians[ksMedians > threshold]

# Выполняем тест Манна-Уитни
stat, p_value = mannwhitneyu(low_sep_group, high_sep_group, alternative='two-sided')

print(f"Тест Манна-Уитни:")
print(f"Статистика: {stat:.4f}")
print(f"p-value: {p_value:.4f}")
if p_value < 0.05:
    print("Различия между группами статистически значимы (p < 0.05)")
else:
    print("Различия между группами не значимы (p >= 0.05)")

# Простая линейная регрессия
slope, intercept, r_value, p_value, std_err = linregress(ksMedians, nnMedians)

# Вычисляем R^2
r_squared = r_value**2

print(f"Линейная регрессия:")
print(f"Наклон (slope): {slope:.4f}")
print(f"Пересечение (intercept): {intercept:.4f}")
print(f"Коэффициент корреляции Пирсона (r): {r_value:.4f}")
print(f"R^2: {r_squared:.4f}")
print(f"p-value: {p_value:.4f}")
if p_value < 0.05:
    print("Связь статистически значима (p < 0.05)")
else:
    print("Связь не значима (p >= 0.05)")

# Визуализация
import matplotlib.pyplot as plt
plt.scatter(ksMedians, nnMedians, color='blue', alpha=0.6)
plt.plot(ksMedians, intercept + slope * ksMedians, color='red', label='Линейная регрессия')
plt.xlabel('Медиана KS')
plt.ylabel('Медиана NN')
plt.title('Связь между разделимостью и точностью (45 пар)')
plt.grid(True)
plt.show()


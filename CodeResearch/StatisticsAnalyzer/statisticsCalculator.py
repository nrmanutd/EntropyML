import matplotlib.pyplot as plt
import numpy as np
from dcor import distance_correlation
from scipy.stats import kendalltau, spearmanr, mannwhitneyu, linregress

from CodeResearch.StatisticsAnalyzer.dataLoader import loadKSData, loadNNData, loadPermutationData, loadData
from CodeResearch.Visualization.metricCalcultation import calculateMetric
from XtendedCorrel import hoeffding


def permutation_test(array1, array2, metric_func, n_permutations=10000):
    """
    Выполняет перестановочный тест для оценки p-value заданной метрики.

    Параметры:
    - array1: первый массив данных (будет перемешиваться)
    - array2: второй массив данных (остается неизменным)
    - metric_func: лямбда-выражение или функция, вычисляющая метрику
    - n_permutations: количество перестановок (по умолчанию 10000)

    Возвращает:
    - observed_stat: истинное значение статистики
    - p_value: p-value по результатам перестановочного теста
    """
    # Вычисляем истинное значение статистики на исходных данных
    observed_stat = metric_func(array1, array2)

    # Счетчик экстремальных значений
    count_extreme = 0

    # Выполняем перестановки
    for _ in range(n_permutations):
        # Случайно перемешиваем первый массив
        permuted_array1 = np.random.permutation(array1)
        # Вычисляем статистику для перестановки
        perm_stat = metric_func(permuted_array1, array2)
        # Считаем, если перестановочная статистика >= наблюдаемой
        if perm_stat >= observed_stat:
            count_extreme += 1

    # Вычисляем p-value как долю экстремальных значений
    p_value = count_extreme / n_permutations

    return observed_stat, p_value

def showStatisticsOverview(testName, value, pvalue):
    valueable = 'значима' if pvalue < 0.05 else 'не значима'
    print(f"Корреляция {testName}: {value:.4f}, p-value: {pvalue:.4f}, связь {valueable}")

def processTaskStatistics(task, directory):
    ksData = task['ksData']
    nnData = task['mlData']
    pData = task['pData']

    taskName = task['taskName']
    print(f'################ Task: {taskName} ################')

    ksMedians = calculateMetric(ksData, pData)
    print(ksMedians)

    if len(ksData) < 2:
        print (f'Мало точек для задачи {taskName}')
        return

    nnMedians = np.array([np.average(k) for k in nnData])
    # ksMedians = np.array([np.quantile(k, upper) - np.quantile(k, lower) for k in ksData[0]])
    # nnMedians = np.array([np.quantile(k, upper) - np.quantile(k, lower) for k in nnData[0]])

    tau, p_value_kendall = kendalltau(ksMedians, nnMedians)
    showStatisticsOverview('Кендалла', tau, p_value_kendall)

    # Для сравнения: Корреляция Спирмена
    corr, p_value_spearman = spearmanr(ksMedians, nnMedians)
    showStatisticsOverview('Спирмена', corr, p_value_spearman)

    threshold = np.median(ksMedians)
    low_sep_group = nnMedians[ksMedians <= threshold]
    high_sep_group = nnMedians[ksMedians > threshold]

    # Выполняем тест Манна-Уитни
    stat, p_value = mannwhitneyu(low_sep_group, high_sep_group, alternative='two-sided')
    showStatisticsOverview('Манна-Уитни', stat, p_value)

    dc_stat, dc_pvalue = permutation_test(ksMedians, nnMedians, lambda a, b: distance_correlation(a, b))
    showStatisticsOverview('Расстояний', dc_stat, dc_pvalue)

    h_stat, h_pvalue = permutation_test(ksMedians, nnMedians, lambda a, b: hoeffding(a, b))
    showStatisticsOverview('Хофдинга', h_stat, h_pvalue)

    # Простая линейная регрессия
    slope, intercept, r_value, p_value, std_err = linregress(ksMedians, nnMedians)

    # Вычисляем R^2
    r_squared = r_value ** 2
    showStatisticsOverview('Линейная регрессия', r_squared, p_value)

    # Визуализация

    plt.scatter(ksMedians, nnMedians, color='blue', alpha=0.6)
    plt.plot(ksMedians, intercept + slope * ksMedians, color='red', label='Linear regression')
    plt.xlabel('KS metric')
    plt.ylabel('ML Accuracy')
    plt.title(f'KS metric and ML accuracy ({len(ksMedians)} classes pairs)')
    plt.grid(True)
    plt.savefig(f'{directory}\\{taskName}_KS_ML_dependency.png', dpi=300, bbox_inches='tight')
    plt.clf()

#directory = "C:\\Current\\Work\\Science\\CodeResearch\\PValuesFigures\\PValueLogs_TargetTasks"
directory = "C:\\Current\\Work\\Science\\CodeResearch\\PValuesFigures\\UCI tasks"
data = loadData(directory)

for k, d in data.items():
    processTaskStatistics(d, directory)
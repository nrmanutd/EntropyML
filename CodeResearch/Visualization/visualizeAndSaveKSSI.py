import re
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from CodeResearch.Visualization.filesExtractor import find_files_with_regex
from CodeResearch.Visualization.saveDataForVisualization import deserialize_labeles_list_of_arrays

def visualizeAndSaveConcretePair(resultsFolder, fitData, alphas, y, taskName, additionalText):
    # Уравнение прямой: log(y) = slope * log(x) + intercept
    # Что эквивалентно: y = exp(intercept) * x^slope

    r_value = fitData[2]
    intercept = fitData[1]
    slope = fitData[0]
    x = alphas

    log_x = np.log(x + 1e-10)
    log_y = np.log(y + 1e-10)

    # Коэффициенты для исходных координат
    A = np.exp(intercept)  # коэффициент перед x^slope
    power = slope  # степень

    # Создаем график
    fig = plt.figure(figsize=(12, 8))

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
             label=f'Прямая: log(y) = {slope:.3f}·log(x) + {intercept:.3f}, k = {2 * slope:.3f}')

    plt.xlabel('log(x)', fontsize=12)
    plt.ylabel('log(y)', fontsize=12)
    plt.title('Логарифмические координаты с линейной регрессией {0} ({1})'.format(taskName, additionalText), fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Добавляем уравнение в виде аннотации
    equation_text = (f'Уравнение в исходных координатах:\n'
                     f'y = {A:.3f}·x^{{{power:.3f}}}\n'
                     f'R² = {r_value ** 2:.4f}')

    plt.figtext(0.5, 0.01, equation_text, ha='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('{:}\\kssi_{:}.png'.format(resultsFolder, taskName),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

def getFitData(x, y):
    # Проверяем, что массивы одинаковой длины
    if len(x) != len(y):
        raise ValueError("Массивы должны быть одинаковой длины")

    # Берем логарифмы (добавляем маленькое значение чтобы избежать log(0))
    log_x = np.log(x + 1e-10)
    log_y = np.log(y + 1e-10)

    # Линейная регрессия в логарифмических координатах
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)

    return slope, intercept, r_value, p_value, std_err

def visualizeAndSaveKSSI(folderWithFiles, resultsFolder, alphas, taskName, iterations):
    if not os.path.exists(resultsFolder):
        os.makedirs(resultsFolder)

    pattern = r"^KS_error_{0}_{1}_\d+_\d+_\d+.txt$".format(taskName, iterations)
    files = find_files_with_regex(folderWithFiles, pattern)

    data = {}
    pattern = r".*_(\d+)_(\d+)_(\d+)\.txt$"
    for file in files:
        match = re.search(pattern, file)
        num1, num2, num3 = match.groups()
        key = "{0}_{1}".format(num1, num2)

        de = deserialize_labeles_list_of_arrays(file)
        value = de[0][-1][0]

        ksFile = file.replace("_error", "_OOS")
        dks = deserialize_labeles_list_of_arrays(ksFile)
        ks = dks[0][-1]

        if key not in data:
            data[key] = {'errors': [], 'objects': [], 'ks': []}

        data[key]['errors'].append(value)
        data[key]['objects'].append(num3)
        data[key]['ks'].append(np.mean(ks))

    x = 1 / np.array(alphas)

    counter = 0
    for k, v in data.items():
        counter += 1
        print('Processing for pair {0} ({1}/{2})'.format(k, counter, len(data)))
        obj = np.array(v['objects'], dtype=np.int32)
        err = np.array(v['errors'], dtype=np.float32)
        ks = np.array(v['ks'])

        idx = np.argsort(obj)
        y = err[idx]
        ks = ks[idx]

        xx = x[1:len(x)]
        y = y[1:len(y)]
        lastks = ks[-1]

        fitData = getFitData(xx, y)
        visualizeAndSaveConcretePair(resultsFolder, fitData, xx, y, "{0}_{1}_{2}".format(taskName, iterations, k), f"ks = {lastks:.2f}")
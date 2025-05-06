import matplotlib.pyplot as plt
import numpy as np
import os

from CodeResearch.Visualization.saveDataForVisualization import deserialize_labeles_list_of_arrays


def plotAndSaveRuntimeSamplesResults(fileName):
    data = deserialize_labeles_list_of_arrays(fileName)
    arrays = data[0]
    labels = data[1]
    taskName = data[2]

    plot_log_dependency(arrays, labels, taskName, 'Log N', f'{os.path.splitext(fileName)[0]}.png')

    pass

def plotAndSaveRuntimeFeaturesResults(fileName):
    data = deserialize_labeles_list_of_arrays(fileName)

    pass


def plot_log_dependency(arrays, labels, taskName, xLabel, output_file="plot.png"):
    """
    Строит график зависимости логарифма средних от логарифма лейблов с линейной регрессией.

    Args:
        arrays (list): Список массивов чисел.
        labels (list): Список строковых лейблов (числа в строковом формате).
        taskName (str): Название задачи для заголовка графика.
        output_file (str): Имя файла для сохранения графика (по умолчанию 'plot.png').
    """
    # Проверяем, что входные данные корректны
    if len(arrays) != len(labels):
        raise ValueError("Количество массивов и лейблов должно совпадать")
    if not arrays or not labels:
        raise ValueError("Списки массивов и лейблов не должны быть пустыми")

    # Преобразуем массивы в numpy для удобства
    arrays = [np.array(arr) for arr in arrays]

    # Вычисляем средние и стандартные отклонения
    means = np.array([np.mean(arr) for arr in arrays])
    stds = np.array([np.std(arr) for arr in arrays])

    # Преобразуем лейблы в числа
    try:
        n_values = np.array([float(label) for label in labels])
    except ValueError as e:
        raise ValueError("Все лейблы должны быть преобразованы в числа") from e

    # Вычисляем логарифмы
    log_n = np.log(n_values)
    log_t = np.log(means)

    # Линейная регрессия
    coeffs = np.polyfit(log_n, log_t, 1)  # [slope, intercept]
    slope, intercept = coeffs
    regression_line = np.poly1d(coeffs)

    # Создаем график
    plt.figure(figsize=(8, 6))

    # Точечный график с погрешностями (стандартное отклонение)
    plt.errorbar(log_n, log_t, yerr=stds / means, fmt='o', color='blue',
                 ecolor='gray', capsize=5, label=f'Log T of {xLabel}')

    # Линия регрессии
    plt.plot(log_n, regression_line(log_n), color='red', linestyle='--',
             label=f'Regression (slope = {slope:.2f})')

    # Настройки графика
    plt.title(f'Log T of {xLabel} for {taskName} dependency')
    plt.xlabel(xLabel)
    plt.ylabel('log T')
    plt.legend()
    plt.grid(True)

    # Сохраняем график
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()
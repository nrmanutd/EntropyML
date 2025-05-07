import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import linregress

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

    arrays = data[0]
    labels = data[1]
    taskName = data[2]

    plot_log_dependency(arrays, labels, taskName, 'Log k', f'{os.path.splitext(fileName)[0]}.png')


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

    funcX = lambda x: np.log(x)
    funcY = lambda y: np.log(y)

    mean_slope, se_slope, ci_lower, ci_upper = bootstrap_slope_ci(
        n_values, np.array(arrays), funcX, funcY, B=500, alpha=0.05, random_state=0
    )

    # Вычисляем логарифмы
    log_n = np.log(n_values)
    log_t = np.log(means)

    # Создаем график
    plt.figure(figsize=(8, 6))

    # Точечный график с погрешностями (стандартное отклонение)
    plt.errorbar(log_n, log_t, yerr=stds / means, fmt='o', color='blue',
                 ecolor='gray', capsize=5, label=f'Log T of {xLabel}')

    # Линейная регрессия
    if len(log_n) > 3:
        coeffs = np.polyfit(log_n, log_t, 1)  # [slope, intercept]
        slope, intercept = coeffs
        regression_line = np.poly1d(coeffs)
        # Линия регрессии
        plt.plot(log_n, regression_line(log_n), color='red', linestyle='--',
             label=f'Regression ({mean_slope:.2f} +/- {se_slope:.2f} [{ci_lower:.2f} - {ci_upper:.2f}])')

    # Настройки графика
    plt.title(f'Log T of {xLabel} for {taskName} dependency')
    plt.xlabel(xLabel)
    plt.ylabel('log T')
    plt.legend()
    plt.grid(True)

    # Сохраняем график
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)

    return

def plot_dependency(arrays, labels, taskName, xLabel, funcX, funcY, funcSelect, output_file="plot.png"):
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
    means = np.array([np.mean(np.array([funcY(k) for k in arr])) for arr in arrays])
    stds = np.array([np.std(np.array([funcY(k) for k in arr])) for arr in arrays])

    # Преобразуем лейблы в числа
    try:
        n_values = np.array([float(label) for label in labels])
    except ValueError as e:
        raise ValueError("Все лейблы должны быть преобразованы в числа") from e

    n_valuesForEstimation = funcSelect(n_values)
    arrsForEstimation = funcSelect(arrays)
    mean_slope, se_slope, ci_lower, ci_upper = bootstrap_slope_ci(
        n_valuesForEstimation, np.array(arrsForEstimation), funcX, funcY, B=500, alpha=0.05, random_state=0
    )

    # Вычисляем логарифмы
    log_n = funcX(n_values)
    log_t = means

    # Создаем график
    plt.figure(figsize=(8, 6))

    # Точечный график с погрешностями (стандартное отклонение)
    plt.errorbar(log_n, log_t, yerr=stds, fmt='o', color='blue',
                 ecolor='gray', capsize=5, label=r'$\log{{t}}$ of '+f'{xLabel}')

    # Линейная регрессия
    if len(log_n) > 3:
        log_n_s = funcSelect(log_n)
        log_t_s = funcSelect(log_t)
        coeffs = np.polyfit(log_n_s, log_t_s, 1)  # [slope, intercept]
        slope, intercept = coeffs
        regression_line = np.poly1d(coeffs)
        # Линия регрессии
        plt.plot(log_n, regression_line(log_n), color='red', linestyle='--',
             label=f'Regression ({mean_slope:.2f}'+r'$\pm$'+f'{se_slope:.2f})')

    # Настройки графика
    plt.title(f'{taskName} ' + r'$\log{{t}}$ of '+f'{xLabel} dependency')
    plt.xlabel(xLabel)
    plt.ylabel(r'$\log{t}$')
    plt.legend()
    plt.grid(True)

    # Сохраняем график
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)

    return

def bootstrap_slope_ci(N_values, times_matrix, funcX, funcY, B=500, alpha=0.05, random_state=None):
    """
    Estimate slope confidence interval on log-log scale via bootstrap.

    Parameters
    ----------
    N_values : array-like, shape (m,)
        Distinct numbers of samples.
    times_matrix : array-like, shape (m, n_runs)
        Recorded runtimes for each N (each row corresponds to one N, columns to repeated runs).
    B : int
        Number of bootstrap replicates.
    alpha : float
        Significance level for the (1 - alpha) confidence interval.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    mean_slope : float
        Bootstrap mean of the slope estimates.
    se_slope : float
        Standard error of the bootstrap slope estimates.
    ci_lower : float
        Lower bound of the (1-alpha) confidence interval.
    ci_upper : float
        Upper bound of the (1-alpha) confidence interval.
    """
    rng = np.random.default_rng(random_state)
    m, n_runs = times_matrix.shape
    logN = funcX(N_values)
    slopes = np.empty(B)

    for b in range(B):
        # Resample runtimes with replacement for each N
        boot_means = []
        for i in range(m):
            sample_indices = rng.integers(0, n_runs, size=n_runs)
            boot_times = times_matrix[i, sample_indices]
            boot_means.append(np.mean(boot_times))
        log_means = funcY(boot_means)

        # Fit a line to (logN, log_means)
        slope, intercept, r_value, p_value, std_err = linregress(logN, log_means)
        slopes[b] = slope

    mean_slope = np.mean(slopes)
    se_slope = np.std(slopes, ddof=1)

    # z-score for two-sided (1-alpha) CI
    z = abs(np.quantile(slopes, [alpha / 2, 1 - alpha / 2]))  # we'll compute CI from percentiles
    ci_lower, ci_upper = np.quantile(slopes, [alpha / 2, 1 - alpha / 2])

    return mean_slope, se_slope, ci_lower, ci_upper
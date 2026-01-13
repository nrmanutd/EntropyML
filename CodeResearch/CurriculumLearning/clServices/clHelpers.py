import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder

from CodeResearch.Helpers.permutationHelpers import stratified_split_indices_with_min
from CodeResearch.LearningFramework.Learners.TorchMLPLearner import TorchMLPLearner
from CodeResearch.LearningFramework.Samplers.RandomWithFixedLengthSampler import RandomWithFixedLengthSampler
from CodeResearch.LearningFramework.Samplers.RandomWithFixedTestSampler import RandomWithFixedTestSampler
from CodeResearch.ObjectComplexity.Diversity.IncrementalObjectDiversifier import IncrementalObjectDiversifier
from CodeResearch.ObjectComplexity.Hardness import ExpandingDatasetHardnessCalculator
from CodeResearch.ObjectComplexity.Hardness.DiversityBasedHardnessCalculator import DiversityBasedHardnessCalculator
from CodeResearch.ObjectComplexity.Hardness.Factory.AssesorEnum import AssesorEnum
from CodeResearch.ObjectComplexity.Hardness.Factory.HardnessFactory import HardnessFactory
from CodeResearch.ObjectComplexity.Hardness.HardnessCorrector import HardnessCorrector
from CodeResearch.ObjectComplexity.Hardness.KSHardnessCalculator import KSHardnessCalculator
from CodeResearch.ObjectComplexity.Hardness.LearnerBasedHardnessCalculator import LearnerBasedHardnessCalculator
from CodeResearch.ObjectComplexity.InstancePriority.multiPrioritiesCalculator import MultiPrioritiesCalculator


def createKSHardnessCalculator(nAttempts, fraction):
    hc = KSHardnessCalculator(nAttempts, fraction)
    hc = ExpandingDatasetHardnessCalculator.ExpandingDatasetHardnessCalculator(hc)
    hc = HardnessCorrector(hc)

    return hc

def createLearnerBasedHardnessCalculator(nAttempts, logger, nFeatures, nClasses, epochs, hidden_sizes, dAttempts, dbatchSize, dEpochs, dHidden_sizes, isLearnerBased: bool):
    l = TorchMLPLearner(input_dim=nFeatures, num_classes=nClasses, hidden_sizes=hidden_sizes, epochs=epochs, update_epochs=dEpochs)
    a = HardnessFactory.createAssesor(AssesorEnum.ShapXGBoost)

    hc = LearnerBasedHardnessCalculator(l, a, nAttempts, logger)

    dcLearner = TorchMLPLearner(input_dim=nFeatures, num_classes=nClasses, hidden_sizes=dHidden_sizes, update_epochs=dEpochs, epochs=dEpochs, batch_size=dbatchSize)
    #hc = DiversityBasedHardnessCalculator(hc, lambda x: NNBasedObjectDiversifier(dcLearner, lambda ds, t: PriorityBasedSampler(ds, t, batchSize, x), dEpochs, logger))
    #hc = DiversityBasedHardnessCalculator(hc, lambda x: SeparableObjectDiversifier(dcLearner, dAttempts, logger))
    hc = DiversityBasedHardnessCalculator(hc, lambda x: IncrementalObjectDiversifier(dcLearner, dAttempts, logger))
    #hc = HardnessCorrector(hc)
    #hc = EasinessInvertor(hc)

    return hc

def createLearnerHC(easinessAttempts, logger, easinessEpochs, diversityAttempts, diversityEpochs, learnerCreator, dataProcessor):
    l = learnerCreator(easinessEpochs)
    a = HardnessFactory.createAssesor(AssesorEnum.ShapXGBoost)

    hc = LearnerBasedHardnessCalculator(l, a, easinessAttempts, dataProcessor, logger)

    dcLearner = learnerCreator(diversityEpochs)
    hc = DiversityBasedHardnessCalculator(hc, lambda x: IncrementalObjectDiversifier(dcLearner, diversityAttempts, 128, dataProcessor, logger))

    return hc

def createSampler(x, y, alphas, betas, testAlpha, repeats, hcBuilder, logger):
    prioritizer = MultiPrioritiesCalculator(hcBuilder, logger, alphas, betas, repeats, True, True, True, False)
    sampler = RandomWithFixedLengthSampler(x, y, prioritizer, 0, testAlpha)

    return sampler

def createSamplerWithTest(x, y, xtest, ytest, alphas, betas, trainAlpha, repeats, hcBuilder, logger):
    prioritizer = MultiPrioritiesCalculator(hcBuilder, logger, alphas, betas, repeats, True, True, True, False)
    sampler = RandomWithFixedTestSampler(x, y, xtest, ytest, prioritizer, trainAlpha, logger)

    return sampler

def processLosses(result):
    arr = np.array(result)
    res = arr.T

    return res

def processEpochLosses(losses):
    result = []

    for loss in losses:
        flat_list = np.concatenate([np.array(sublist) for sublist in loss])
        result.append(flat_list)

    return result

def filterDataSet(x, y, alpha, firstClass, secondClass):
    enc = LabelEncoder()
    target = enc.fit_transform(np.ravel(y))

    firstObjects = np.where(target == firstClass)[0]
    secondObjects = np.where(target == secondClass)[0]

    idx = list(set(firstObjects) | set(secondObjects))
    firstK = math.ceil(alpha * len(idx))
    idx = idx[:firstK]

    tt = enc.fit_transform(np.ravel(target[idx]))

    return x[idx], tt

def filterDataSetByFraction(x, y, alpha):
    enc = LabelEncoder()
    target = enc.fit_transform(np.ravel(y))

    idx, restIdx = stratified_split_indices_with_min(target, alpha)

    tt = enc.fit_transform(np.ravel(target[idx]))

    return x[idx], tt


def visualizeAndSaveComplexity(easiness, importance, scores, xLabel, yLabel, title, filename):
    """
    Создает график точек по координатам и сохраняет в файл

    Parameters:
    easiness (list): массив x-координат (горизонтальная ось)
    importance (list): массив y-координат (вертикальная ось)
    filename (str): путь и имя файла для сохранения
    """
    # Проверяем, что массивы одинаковой длины
    if len(easiness) != len(importance):
        raise ValueError("Массивы должны быть одинаковой длины")

    # Создаем график
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(easiness, importance,
                          c=scores,  # массив скоров для цвета
                          cmap='viridis',  # цветовая карта
                          alpha=0.7,
                          s=50,
                          edgecolor='k',  # черная обводка точек
                          linewidth=0.5)

    # Добавляем colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Scores', fontsize=12)

    # Настраиваем оси и заголовок
    plt.xlabel(xLabel, fontsize=12)
    plt.ylabel(yLabel, fontsize=12)

    r = check_independence(easiness, importance)
    plt.title(f'{title} \n pearson: {r['pearson']['r']: .2f}, {r['pearson']['p']: .2f} spearman: {r['spearman']['r']: .2f}, {r['spearman']['p']: .2f}  mi: {r['mutual_info']: .2f}, any: {r['any_dependence']}', fontsize=14)

    # Добавляем сетку для лучшей читаемости
    plt.grid(True, alpha=0.3)

    # Сохраняем график
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Закрываем график для освобождения памяти


def check_independence(x, y):
    """Проверяет зависимость тремя основными тестами"""
    x, y = np.array(x), np.array(y)

    # 1. Pearson (линейная)
    pearson_r, pearson_p = stats.pearsonr(x, y)

    # 2. Spearman (монотонная)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    # 3. Mutual Information (нелинейная)
    mi = mutual_info_regression(x.reshape(-1, 1), y)[0]

    return {
        'pearson': {'r': pearson_r, 'p': pearson_p},
        'spearman': {'r': spearman_r, 'p': spearman_p},
        'mutual_info': mi,
        'any_dependence': (pearson_p < 0.05 or spearman_p < 0.05 or mi > 0.1)
    }

def plot_complexity_distributions(easiness, importance, filename):
    """
    Создает два графика распределений (один под другим) для easiness и importance

    Parameters:
    easiness (list): массив значений easiness
    importance (list): массив значений importance
    filename (str): путь и имя файла для сохранения
    """
    # Проверяем, что массивы одинаковой длины
    if len(easiness) != len(importance):
        raise ValueError("Массивы должны быть одинаковой длины")

    # Создаем фигуру с двумя subplots (один под другим)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # График распределения для easiness
    ax1.hist(easiness, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Easiness', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Easiness', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Добавляем статистическую информацию
    mean_easiness = np.mean(easiness)
    std_easiness = np.std(easiness)
    ax1.axvline(mean_easiness, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_easiness:.2f}')
    ax1.legend()

    # График распределения для importance
    ax2.hist(importance, bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('Importance', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Importance', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Добавляем статистическую информацию
    mean_importance = np.mean(importance)
    std_importance = np.std(importance)
    ax2.axvline(mean_importance, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_importance:.2f}')
    ax2.legend()

    # Настраиваем расстояние между subplots
    plt.tight_layout()

    # Сохраняем график
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def filter_data(easiness, importance):
    """
    Фильтрует данные, удаляя NaN и бесконечные значения
    """
    # Преобразуем в numpy arrays
    easiness = np.array(easiness, dtype=float)
    importance = np.array(importance, dtype=float)

    # Создаем маску для валидных данных
    mask = np.isfinite(easiness) & np.isfinite(importance)

    # Подсчитываем количество отфильтрованных точек
    filtered_count = len(easiness) - np.sum(mask)
    if filtered_count > 0:
        print(f"Предупреждение: отфильтровано {filtered_count} невалидных точек")

    return easiness[mask], importance[mask]

def plot_distributions_kde_with_metrics(easiness, importance, filename):
    """
    Версия с метриками сравнения реального распределения и идеальной гауссианы
    """
    if len(easiness) != len(importance):
        raise ValueError("Массивы должны быть одинаковой длины")

    easiness, importance = filter_data(easiness, importance)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 11))

    # Easiness
    kde_easiness = stats.gaussian_kde(easiness)
    x_easiness = np.linspace(min(easiness), max(easiness), 100)
    ax1.plot(x_easiness, kde_easiness(x_easiness),
             color='blue', linewidth=2, label='KDE')


    ax1.fill_between(x_easiness, kde_easiness(x_easiness),
                     alpha=0.3, color='skyblue')
    ax1.set_xlabel('Easiness', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Easiness Distribution (KDE)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Importance с расширенной информацией
    kde_importance = stats.gaussian_kde(importance)
    variance_importance = np.var(importance)
    std_importance = np.sqrt(variance_importance)
    mean_importance = np.mean(importance)

    # Диапазон для графиков
    x_min = min(min(importance), -4 * std_importance)
    x_max = max(max(importance), 4 * std_importance)
    x_combined = np.linspace(x_min, x_max, 400)

    # Реальное распределение
    real_pdf = kde_importance(x_combined)
    ax2.plot(x_combined, real_pdf,
             color='red', linewidth=2.5, label='Реальное распределение (KDE)')
    ax2.fill_between(x_combined, real_pdf, alpha=0.3, color='lightcoral')

    # Идеальная гауссиана
    ideal_gaussian = stats.norm(loc=0, scale=std_importance)
    ideal_pdf = ideal_gaussian.pdf(x_combined)
    ax2.plot(x_combined, ideal_pdf,
             color='green', linewidth=2, linestyle='--',
             label='Идеальная гауссиана')

    # Вычисляем расхождение между распределениями (KL divergence approximation)
    # Используем только точки где оба PDF > 0
    mask = (real_pdf > 1e-6) & (ideal_pdf > 1e-6)
    if np.sum(mask) > 10:
        kl_divergence = np.sum(real_pdf[mask] * np.log(real_pdf[mask] / ideal_pdf[mask])) * (
                    x_combined[1] - x_combined[0])
        kl_text = f'KL divergence: {kl_divergence:.3f}'
    else:
        kl_text = 'KL divergence: N/A'

    # Добавляем текстовую информацию
    stats_text = (f'Реальное распределение:\n'
                  f'μ = {mean_importance:.2f}, σ = {std_importance:.2f}\n'
                  f'Идеальная гауссиана:\n'
                  f'μ = 0, σ = {std_importance:.2f}\n'
                  f'{kl_text}')

    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax2.set_xlabel('Importance', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Сравнение распределения Importance с идеальной гауссианой', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_object_metrics(metric_arrays, title, fileName):
        """
        Строит график метрик для каждого объекта в зависимости от номера расчета.

        Параметры:
        ----------
        metric_arrays : list of numpy arrays or list of lists
            Список массивов, где каждый массив содержит значения метрик
            для всех объектов в конкретном расчете.
            Формат: [array_k1, array_k2, ..., array_kl]
            где array_ki содержит n значений для i-го расчета
        """
        # Преобразуем в numpy массив для удобства
        metrics_matrix = np.array(metric_arrays)

        # Проверяем, что все массивы имеют одинаковую длину
        if len(set(len(arr) for arr in metric_arrays)) > 1:
            raise ValueError("Все массивы должны иметь одинаковую длину")

        # Количество расчетов (k) и количество объектов (n)
        l, n = metrics_matrix.shape

        # Создаем график
        plt.figure(figsize=(10, 6))

        # Для каждого объекта строим линию
        for obj_idx in range(n):
            # Получаем значения метрик для данного объекта по всем расчетам
            object_metrics = metrics_matrix[:, obj_idx]

            # Номера расчетов (ось X)
            calculation_numbers = np.arange(1, l + 1)

            # Строим график для данного объекта
            plt.plot(calculation_numbers, object_metrics,
                     marker='o', label=f'Объект {obj_idx + 1}')

        # Настройка графика
        plt.xlabel('Номер расчета (k)', fontsize=12)
        plt.ylabel('Значение величины', fontsize=12)
        plt.title(f'Динамика метрик для {n} объектов по {l} расчетам {title}', fontsize=14)
        plt.grid(True, alpha=0.3)
        #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Устанавливаем целые значения на оси X
        plt.xticks(np.arange(1, l + 1))

        plt.tight_layout()
        plt.savefig(fileName, dpi=300, bbox_inches='tight')
        plt.close()


def plot_distributions_kde(metric_arrays, fileName,
                           labels=None,
                           title="Распределения значений по расчетам",
                           xlabel="Значение величины",
                           ylabel="Плотность вероятности",
                           figsize=(12, 7),
                           colors=None,
                           linestyles=None,
                           alpha=0.7,
                           linewidth=2,
                           bandwidth=None,
                           show_points=False,
                           fill_under=False):
    """
    Строит график сглаженных распределений (KDE) для каждого массива расчетов.

    Параметры:
    ----------
    metric_arrays : list of arrays
        Список массивов, где каждый массив содержит значения для всех объектов
        в конкретном расчете (k-й расчет)
    labels : list of str, optional
        Подписи для каждого распределения в легенде.
        По умолчанию: "Расчет 1", "Расчет 2", ...
    title : str, optional
        Заголовок графика
    xlabel, ylabel : str, optional
        Подписи осей
    figsize : tuple, optional
        Размер графика
    colors : list, optional
        Список цветов для линий (один цвет на каждое распределение)
    linestyles : list, optional
        Список стилей линий
    alpha : float, optional
        Прозрачность линий
    linewidth : int, optional
        Толщина линий
    bandwidth : float or str, optional
        Параметр сглаживания для KDE. Если None, используется правило Скотта.
        Можно указать число или 'scott', 'silverman'
    show_points : bool, optional
        Если True, показывает отдельные точки (rug plot) под каждым распределением
    fill_under : bool, optional
        Если True, заполняет область под кривыми
    """
    # Проверяем, что все массивы имеют данные
    for i, arr in enumerate(metric_arrays):
        if len(arr) == 0:
            raise ValueError(f"Массив {i} пустой")

    # Количество расчетов
    l = len(metric_arrays)

    # Создаем подписи по умолчанию
    if labels is None:
        labels = [f'Расчет {i + 1}' for i in range(l)]

    # Генерируем цвета по умолчанию
    if colors is None:
        cmap = plt.cm.get_cmap('tab10', l)
        colors = [cmap(i) for i in range(l)]

    # Стили линий по умолчанию
    if linestyles is None:
        linestyles = ['-', '--', '-.', ':'] * ((l // 4) + 1)
        linestyles = linestyles[:l]

    # Создаем график
    fig, ax = plt.subplots(figsize=figsize)

    # Для каждого массива строим KDE
    all_data = []
    for i, arr in enumerate(metric_arrays):
        data = np.array(arr)
        all_data.append(data)

        # Строим KDE
        if bandwidth is None or isinstance(bandwidth, str):
            kde = stats.gaussian_kde(data, bw_method=bandwidth)
        else:
            kde = stats.gaussian_kde(data)
            kde.set_bandwidth(bw_method=bandwidth)

        # Определяем диапазон для построения графика
        min_val = data.min()
        max_val = data.max()
        x_range = np.linspace(min_val - (max_val - min_val) * 0.1,
                              max_val + (max_val - min_val) * 0.1, 1000)

        # Вычисляем значения KDE
        y = kde(x_range)

        # Строим график
        if fill_under:
            ax.fill_between(x_range, y, alpha=alpha * 0.3, color=colors[i])

        ax.plot(x_range, y,
                label=labels[i],
                color=colors[i],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=linewidth,
                alpha=alpha)

        # Показываем точки если нужно
        if show_points:
            # Rug plot - маленькие вертикальные линии под осью x
            ax.plot(data, np.zeros_like(data) - 0.01,
                    '|', color=colors[i], alpha=0.5, markersize=10)

    # Настройка графика
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{title} (l={l} расчетов)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # Автоматический подбор пределов осей
    all_values = np.concatenate(all_data)
    x_min, x_max = all_values.min(), all_values.max()
    x_padding = (x_max - x_min) * 0.05
    ax.set_xlim(x_min - x_padding, x_max + x_padding)

    plt.tight_layout()

    plt.savefig(fileName, dpi=300, bbox_inches='tight')
    plt.close()

    return fig, ax


def plot_multiple_ecdfs(ecdfs_list, intersections = None, labels=None, title="Multiple ECDFs",
                        fileName="multiple_ecdfs.png", figsize=(12, 8)):
    """
    Строит несколько ECDF на одном графике

    Parameters:
    -----------
    ecdfs_list : list of tuples
        Список из (x, y) пар для каждой ECDF
        x - отсортированные значения выборки
        y - соответствующие значения вероятностей (от 0 до 1)

    labels : list of str, optional
        Подписи для каждой кривой ECDF

    title : str
        Заголовок графика

    filename : str
        Имя файла для сохранения

    figsize : tuple
        Размер фигуры (ширина, высота)
    """
    # Проверка входных данных
    if not ecdfs_list:
        raise ValueError("Список ECDF не должен быть пустым")

    n_ecdfs = len(ecdfs_list)

    # Автоматические подписи, если не заданы
    if labels is None:
        labels = [f"Выборка {i + 1}" for i in range(n_ecdfs)]
    elif len(labels) != n_ecdfs:
        raise ValueError(f"Количество подписей ({len(labels)}) должно совпадать "
                         f"с количеством ECDF ({n_ecdfs})")

    # Создаем цветовую карту
    cmap = plt.get_cmap('tab10') if n_ecdfs <= 10 else plt.get_cmap('tab20')
    colors = [cmap(i % cmap.N) for i in range(n_ecdfs)]

    # Создаем график
    plt.figure(figsize=figsize)

    if intersections is not None and len(intersections) > 0:
        intersections = np.asarray(intersections)
        x_intersections = intersections[:]

        # Рисуем точки пересечения
        plt.scatter(x_intersections, x_intersections,
                    marker='X', s=150, c='red',
                    edgecolors='black', linewidths=2, zorder=10)

    # Рисуем все ECDF
    all_x_min = []
    all_x_max = []

    for i, (x, y) in enumerate(ecdfs_list):
        # Преобразуем в numpy массивы

        x = np.asarray(x)
        y = np.asarray(y)

        plt.step(x, y, where='post',
                 label=labels[i],
                 color=colors[i],
                 linewidth=2,
                 alpha=0.8)

        # Запоминаем min и max для установки границ
        all_x_min.append(x.min())
        all_x_max.append(x.max())

    # Настраиваем график
    plt.xlabel('Значение', fontsize=12)
    plt.ylabel('F(x) - Эмпирическая функция распределения', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')

    # Устанавливаем границы по X с небольшим запасом
    x_min = min(all_x_min) - 0.1 * (max(all_x_max) - min(all_x_min))
    x_max = max(all_x_max) + 0.1 * (max(all_x_max) - min(all_x_min))
    plt.xlim(x_min, x_max)

    # Y всегда от 0 до 1
    plt.ylim(-0.05, 1.05)

    # Легенда
    plt.legend(loc='lower right', fontsize=10, framealpha=0.9)

    # Сетка
    plt.grid(True, alpha=0.3, linestyle='--')

    # Сохраняем с высоким качеством
    plt.tight_layout()
    plt.savefig(fileName, dpi=300, bbox_inches='tight')
    plt.close()

def ecdf_advanced(data):
    """
    ECDF с правильной обработкой повторяющихся значений
    """
    data = np.asarray(data)
    n = len(data)

    # Сортируем
    x = np.sort(data)

    # Для повторяющихся значений нужно корректно считать
    # Используем cumcount
    y = np.arange(1, n + 1) / n

    # Если есть NaN, удаляем их
    if np.any(np.isnan(data)):
        valid = ~np.isnan(data)
        x = x[valid]
        n = len(x)
        y = np.arange(1, n + 1) / n

    return x, y


def find_ecdf_intersection_optimize(x1, y1, x2, y2):
    """
    Использование оптимизации для поиска пересечения.
    """
    # Создаем интерполяционные функции
    f1 = interpolate.interp1d(x1, y1, kind='linear',
                              bounds_error=False, fill_value=(0, 1))
    f2 = interpolate.interp1d(x2, y2, kind='linear',
                              bounds_error=False, fill_value=(0, 1))

    # Функция разности
    def diff_func(x):
        return f1(x) - f2(x)

    # Диапазон поиска (минимальный и максимальный x из обоих наборов)
    x_min = min(x1.min(), x2.min())
    x_max = max(x1.max(), x2.max())

    # Ищем корни разности функций
    intersections = []

    # Разбиваем на интервалы для поиска
    search_points = np.linspace(x_min, x_max, 100)

    for i in range(len(search_points) - 1):
        a, b = search_points[i], search_points[i + 1]

        # Проверяем, меняет ли функция знак на интервале
        if diff_func(a) * diff_func(b) < 0:
            intersections.append(a)

    return intersections
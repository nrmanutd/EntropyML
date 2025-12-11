import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder

from CodeResearch.LearningFramework.Learners.TorchLearner import TorchMLPLearner
from CodeResearch.LearningFramework.Samplers.RandomWithFixedLengthSampler import RandomWithFixedLengthSampler
from CodeResearch.ObjectComplexity.Hardness import ExpandingDatasetHardnessCalculator
from CodeResearch.ObjectComplexity.Hardness.KSHardnessCalculator import KSHardnessCalculator
from CodeResearch.ObjectComplexity.Hardness.LearnerBasedHardnessCalculator import LearnerBasedHardnessCalculator
from CodeResearch.ObjectComplexity.InstancePriority.multiPrioritiesCalculator import MultiPrioritiesCalculator
from CodeResearch.ObjectComplexity.ObjectAssessment.StandardAssesor import StandardAssesor


def createKSHardnessCalculator(nAttempts, fraction):
    hc = KSHardnessCalculator(nAttempts, fraction)
    hc = ExpandingDatasetHardnessCalculator.ExpandingDatasetHardnessCalculator(hc)
    return hc

def createLearnerBasedHardnessCalculator(nAttempts, fraction, logger, nFeatures, nClasses):
    hardnessLearner = TorchMLPLearner(input_dim=2 * nFeatures, num_classes=nClasses, hidden_sizes=(16, 16))
    assesor = StandardAssesor()

    hc = LearnerBasedHardnessCalculator(hardnessLearner, assesor, nAttempts, fraction, logger)
    hc = ExpandingDatasetHardnessCalculator.ExpandingDatasetHardnessCalculator(hc)
    return hc

def calculateLosses(x, y, alphas, betas, testAlpha, repeats, generalLearner, learner, hc):

    prioritizer = MultiPrioritiesCalculator(hc, alphas, betas, repeats, True, True, True, True)

    sampler = RandomWithFixedLengthSampler(x, y, prioritizer, 0, testAlpha)

    result = generalLearner.estimateLearner(sampler, learner)
    return result

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

    return x[idx, :], tt


def visualizeAndSaveComplexity(easiness, importance, filename):
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
    plt.scatter(easiness, importance, alpha=0.7, s=50)

    # Настраиваем оси и заголовок
    plt.xlabel('Easiness', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.title('Easiness vs Importance', fontsize=14)

    # Добавляем сетку для лучшей читаемости
    plt.grid(True, alpha=0.3)

    # Сохраняем график
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Закрываем график для освобождения памяти


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
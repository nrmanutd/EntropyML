import math
import matplotlib.pyplot as plt

import numpy as np
from sklearn.preprocessing import LabelEncoder

from CodeResearch.LearningFramework.Samplers.RandomWithFixedLengthSampler import RandomWithFixedLengthSampler
from CodeResearch.ObjectComplexity.Hardness import ExpandingDatasetHardnessCalculator
from CodeResearch.ObjectComplexity.Hardness.HardnessCalculator import HardnessCalculator
from CodeResearch.ObjectComplexity.InstancePriority.multiPrioritiesCalculator import MultiPrioritiesCalculator


def calculateLosses(x, y, alphas, testAlpha, nAttempts, fraction, generalLearner, learner):
    hc = HardnessCalculator(nAttempts, fraction)
    hardnessCalculator = ExpandingDatasetHardnessCalculator.ExpandingDatasetHardnessCalculator(hc)

    prioritizer = MultiPrioritiesCalculator(hardnessCalculator, alphas, True, True, True, True)

    sampler = RandomWithFixedLengthSampler(x, y, prioritizer, 0, testAlpha)

    result =  generalLearner.estimateLearner(sampler, learner)

    arr = np.array(result)
    res = arr.T

    return res

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
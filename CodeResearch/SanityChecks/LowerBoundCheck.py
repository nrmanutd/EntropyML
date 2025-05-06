import math
import time
from sklearn import datasets
from scipy import stats
from sklearn.utils import resample

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import LabelEncoder

from CodeResearch.DiviserCalculation.diviserHelpers import get_one_bit_indices
from CodeResearch.DiviserCalculation.getDiviserFastNumba import getMaximumDiviserFastNumba
from CodeResearch.DiviserCalculation.getDiviserTrueKS import getMaximumDiviserTrueKS
from CodeResearch.calcModelEstimations import calcXGBoost, calcLinRegression
from CodeResearch.dataSets import make_spirals, make_xor, make_random
from CodeResearch.pValueCalculator import getDataSetIndexesOfTwoClasses


def getSubset(currentObjects, dataSet, target, iClass, jClass):

    allIdxes = np.array(range(len(target)))
    iClassIdx, jClassIdx = getDataSetIndexesOfTwoClasses(currentObjects, target, iClass, jClass)
    idx = np.concatenate((iClassIdx, jClassIdx))

    testIdx = np.setdiff1d(allIdxes, idx)

    tClasses = target[idx]
    dsClasses = dataSet[idx, :]

    testDs = dataSet[testIdx, :]
    testTClasses = target[testIdx]

    return dsClasses, tClasses, testDs, testTClasses

def calcFastKS(dataSet, target):
    return getMaximumDiviserFastNumba(dataSet, target)[0]



def calcFastRestartsKS(dataSet, target, n_restarts=8, rng=None):
    X = dataSet
    nFeatures = dataSet.shape[1]
    rng = np.random.default_rng(rng)
    best_val = -np.inf

    if 2**nFeatures < n_restarts:
        n_restarts = 2**(nFeatures - 1)

    selected = np.random.choice(np.arange(2**nFeatures), size=n_restarts, replace=False)

    for _ in selected:
        indices = get_one_bit_indices(_)
        signs = np.ones(nFeatures)
        signs[indices] = -1
        X_flip = X * signs  # отражаем точки
        val = getMaximumDiviserFastNumba(X_flip, target)[0]

        if val > best_val:
            best_val = val
    return best_val

def calcTrueKS(dataSet, target):
    return getMaximumDiviserTrueKS(dataSet, target)


def compareKSandFastKS(dataSet, target, attempts):
    trueKS = []
    fastKS = []
    xgBoost = []

    nObjects = dataSet.shape[0]

    enc = LabelEncoder()
    target = enc.fit_transform(np.ravel(target))

    for i in range(attempts):
        t1 = time.time()

        ds, t, testDs, testT = getSubset(math.floor(nObjects / 2), dataSet, target, 0, 1)

        #fastValue = calcFastKS(ds, t)
        fastValue = calcFastRestartsKS(ds, t)
        trueValue = calcTrueKS(ds, t)
        #accuracy = calcLinRegression(ds, t, testDs, testT)

        fastKS.append(fastValue)
        trueKS.append(trueValue)
        #xgBoost.append(accuracy)

        print(f'Attempt #{i} of {attempts}, time: {time.time() - t1} s')

    return np.array(trueKS), np.array(fastKS), np.array(xgBoost)

def show(trueKS, fastKS, ml, taskName):
    vector1 = trueKS
    vector2 = fastKS
    vector3 = ml
    # Вычисление корреляции Пирсона
    corr_pearson, p_value_pearson = pearsonr(vector1, vector2)

    # Вычисление корреляции Спирмена
    corr_spearman, p_value_spearman = spearmanr(vector1, vector2)

    delta = np.mean(vector1 - vector2)

    # Вывод результатов
    print(f"Среднее отклонение: {delta:.3f}")
    print(f"Корреляция Пирсона: {corr_pearson:.3f}, p-value: {p_value_pearson:.3f}")
    print(f"Корреляция Спирмена: {corr_spearman:.3f}, p-value: {p_value_spearman:.3f}")

    # Визуализация
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(vector1, vector2, color='blue', label='True KS vs Fast KS')
    #plt.scatter(vector1, ml, color='red', label='ML')
    plt.scatter(vector1, vector1, color='purple', label='Ideal line')

    # Добавляем линию регрессии
    z = np.polyfit(vector1, vector2, 1)  # Линейная регрессия (степень 1)
    p = np.poly1d(z)
    plt.plot(vector1, p(vector1), color='red', linestyle='--', label='Regression')

    # Настройки графика
    plt.title(f'True KS vs Fast KS relation ship for {taskName}')
    plt.xlabel('True KS')
    plt.ylabel('Fast KS')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Показываем график

    plt.savefig(f'{taskName}_trueKS_vs_fastKS.png',dpi=300, bbox_inches='tight')
    plt.close(fig)


def ks_approx_metrics(v_true, v_pred, n_boot=10_000, random_state=42):
    """
    v_true : 1‑D array‑like – «честные» (точные) значения KS‑статистики
    v_pred : 1‑D array‑like – значения приближённой метрики (G‑SOMKS)
    n_boot : int          – размер бутстрепа для 95 % CI Pearson‑R
    """
    v_true = np.asarray(v_true, dtype=np.float64)
    v_pred = np.asarray(v_pred, dtype=np.float64)

    if v_true.shape != v_pred.shape:
        raise ValueError("v_true and v_pred must have the same shape")

    delta = v_true - v_pred  # ошибки приближения
    abs_delta = np.abs(delta)

    # --- базовые ошибки ------------------------------------------------------
    bias = delta.mean()
    rmse = np.sqrt((delta ** 2).mean())
    rel_error = rmse / v_true.mean()  # относительная RMSE
    med_abs = np.median(abs_delta)
    p90_abs = np.percentile(abs_delta, 90)

    # --- корелляции ----------------------------------------------------------
    spearman_r, spearman_p = stats.spearmanr(v_true, v_pred)
    kendall_t, kendall_p = stats.kendalltau(v_true, v_pred)
    pearson_r, pearson_p = stats.pearsonr(v_true, v_pred)
    r2 = pearson_r ** 2

    # --- бутстреп 95 % CI для Pearson‑R --------------------------------------
    rng = np.random.default_rng(random_state)
    boot_stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(v_true), len(v_true))
        r, _ = stats.pearsonr(v_true[idx], v_pred[idx])
        boot_stats.append(r)
    ci_low, ci_high = np.percentile(boot_stats, [2.5, 97.5])

    # --- собрать результаты ---------------------------------------------------
    return {
        "bias": bias,
        "rmse": rmse,
        "relative_error": rel_error,
        "median_|Δ|": med_abs,
        "90pct_|Δ|": p90_abs,
        "spearman_rho": spearman_r,
        "spearman_p": spearman_p,
        "kendall_tau": kendall_t,
        "kendall_p": kendall_p,
        "pearson_r": pearson_r,
        "pearson_r2": r2,
        "pearson_p": pearson_p,
        "pearson_r_CI95%": (ci_low, ci_high),
    }

# пример использования --------------------------------------------------------
# v1 = ...  # «честный» KS
# v2 = ...  # приближение
# metrics = ks_approx_metrics(v1, v2)
# for k, v in metrics.items():
#     print(f"{k:>15}: {v}")


def printData(metrics, taskName):
    # аккуратный вывод -------------------------------------------------
    print(f"\n===  G‑SOMKS vs exact MKS sanity check for {taskName}  ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:>15}: {v: .6f}")
        else:  # кортеж доверительного интервала
            low, high = v
            print(f"{k:>15}: [{low: .6f}, {high: .6f}]")

def calculateTrueVsFastKS(x, y, taskName):
    attempts = 100

    print(f'Task name: {taskName}')
    trueKS, fastKS, xgBoost = compareKSandFastKS(x, y, attempts)
    print(trueKS)
    print(fastKS)
    print(xgBoost)

    result = ks_approx_metrics(trueKS, fastKS)
    printData(result, taskName)

    #result = ks_approx_metrics(trueKS, xgBoost)
    #printData(result)

    #result = ks_approx_metrics(fastKS, xgBoost)
    #printData(result)

    show(trueKS, fastKS, xgBoost, taskName)
    return

nSamples = 200

x, y = datasets.make_blobs(n_samples=nSamples, centers=2, n_features=2, random_state=42)
calculateTrueVsFastKS(x, y, 'blobs_1')

x, y = make_xor(nSamples)
calculateTrueVsFastKS(x, y, 'xor')

x, y = datasets.make_circles(n_samples=nSamples, factor=0.5, noise=0.1, random_state=42)
calculateTrueVsFastKS(x, y, 'circles')

x, y = datasets.make_blobs(n_samples=nSamples, centers=2, n_features=2, random_state=42, cluster_std=9)
calculateTrueVsFastKS(x, y, 'blobs_9')

x, y = datasets.make_moons(n_samples=nSamples, noise=0.1, random_state=42)
calculateTrueVsFastKS(x, y, 'moons')

x, y = make_spirals(nSamples)
calculateTrueVsFastKS(x, y, 'spirals')

x, y = make_random(nSamples)
calculateTrueVsFastKS(x, y, 'random')


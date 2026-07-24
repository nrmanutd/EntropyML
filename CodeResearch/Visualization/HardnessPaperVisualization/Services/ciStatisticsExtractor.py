import numpy as np
from typing import Tuple, Optional

from CodeResearch.Visualization.HardnessPaperVisualization.Services.extractData import extractTask, extractFilesForParameters
from CodeResearch.Visualization.saveDataForVisualization import deserialize_labeles_list_of_arrays

def extractArea(arr):
    areas = []
    for i in range(len(arr[0])):
        curArea = 0
        for j in range(len(arr)):
            curArea += arr[j][i] / len(arr)

        areas.append(curArea)

    return np.array(areas)

def extractAccuracy(arr):
    accuracies = []
    for i in range(len(arr[0])):
        bestAccuracy = 0
        for j in range(len(arr)):
            if bestAccuracy < arr[j][i]:
                bestAccuracy = arr[j][i]

        accuracies.append(bestAccuracy)

    return np.array(accuracies)

def calculateCI(
    ours: np.ndarray,
    best_baseline: np.ndarray,
    percentile: float = 95.0,
    n_bootstrap: int = 10_000,
    random_state: Optional[int] = 0,
    ) -> Tuple[float, float, float]:
    """
    Bootstrap CI for the difference in means: E[ours] - E[best_baseline].


    Args:
    ours: 1D array of metric values from our method across runs (unpaired).
    best_baseline: 1D array of metric values from the best baseline across runs (unpaired).
    percentile: CI level in percent, e.g. 95.0 for a 95% CI.
    n_bootstrap: Number of bootstrap resamples (standard choice: 10k).
    random_state: Seed for reproducibility (None -> non-deterministic).


    Returns:
    (ci_low, ci_high): percentile bootstrap CI bounds for mean(ours)-mean(baseline).
    """
    ours = np.asarray(ours, dtype=float).ravel()
    best_baseline = np.asarray(best_baseline, dtype=float).ravel()

    if ours.size < 2 or best_baseline.size < 2:
        raise ValueError("Both inputs must contain at least 2 samples.")
    if not (0.0 < percentile < 100.0):
        raise ValueError("percentile must be in (0, 100).")


    rng = np.random.default_rng(random_state)
    n1, n2 = ours.size, best_baseline.size
    delta = float(ours.mean() - best_baseline.mean())

    # Bootstrap with replacement within each group (unpaired bootstrap)
    idx1 = rng.integers(0, n1, size=(n_bootstrap, n1))
    idx2 = rng.integers(0, n2, size=(n_bootstrap, n2))

    diff_boot = ours[idx1].mean(axis=1) - best_baseline[idx2].mean(axis=1)

    alpha = 1.0 - percentile / 100.0
    lo = np.quantile(diff_boot, alpha / 2.0)
    hi = np.quantile(diff_boot, 1.0 - alpha / 2.0)
    return delta, float(lo), float(hi)

def extractFunctionalCI(folder, protocol, task, fraction, p, valueExtractor):
    r = extractTask(folder, task.lower())

    files = extractFilesForParameters(r, fraction, protocol)

    targetAreas = None
    baseLines = []

    for file in files:
        rr = deserialize_labeles_list_of_arrays(file['file'])
        arrays = rr[0]

        if file['mode'] == 'h&i_inc':
            targetAreas = valueExtractor(arrays)
        else:
            baseLines.append(valueExtractor(arrays))

    currentBaseLine = None
    bestBaseLine = None
    for i in range(len(baseLines)):
        cur = np.mean(targetAreas) - np.mean(baseLines[i])
        if currentBaseLine is None:
            currentBaseLine = cur
            bestBaseLine = baseLines[i]
        else:
            if currentBaseLine > cur:
                currentBaseLine = cur
                bestBaseLine = baseLines[i]

    mean, left, right = calculateCI(targetAreas, bestBaseLine, p)
    return mean, left, right

def extractAulcCI(folder, protocol, task, fraction, p):
    return extractFunctionalCI(folder, protocol, task, fraction, p, lambda x: extractArea(x))

def extractAccCI(folder, protocol, task, fraction, p):
    return extractFunctionalCI(folder, protocol, task, fraction, p, lambda x: extractAccuracy(x))

def saveCIToFile(accTable, aulcTable, tasks, fileName):
    accs = accTable.shape
    aulcs = aulcTable.shape
    accm = 100
    aulcm = 100

    for i in range(accs[0]):
        with open(f'{i}_{fileName}', 'w', encoding='utf=8') as file:
            file.write('\\multirow{3}{*}{\\rotatebox[origin=c]{90}{ACC}}\n')
            for j in range(accs[1]):
                line = f'& {tasks[j]} '

                for k in range(accs[2]):
                    sign = '+' if accTable[i, j, k, 0] > 0 else ''
                    cc = accTable[i, j, k, 0]*accm
                    ll = accTable[i, j, k, 1]*accm
                    rr = accTable[i, j, k, 2]*accm

                    c = f'{cc:.2f}'

                    if sign == '+':
                        c = '\\textbf{' + f'{sign}{c}' + '}'

                    line += f'&{c} $\\pm$ {(rr-ll)/2:.0e}'
                    if k == accs[2] - 1:
                        line += '\\\\\n'
                file.write(line)

            file.write('\\midrule\n')
            file.write('\\multirow{3}{*}{\\rotatebox[origin=c]{90}{AULC}}\n')

            for j in range(aulcs[1]):
                line = f'& {tasks[j]} '

                for k in range(aulcs[2]):
                    sign = '+' if aulcTable[i, j, k, 0] > 0 else ''
                    cc = aulcTable[i, j, k, 0] * aulcm
                    ll = aulcTable[i, j, k, 1] * aulcm
                    rr = aulcTable[i, j, k, 2] * aulcm

                    c = f'{cc:.2f}'

                    if sign == '+':
                        c = '\\textbf{' + f'{sign}{c}' + '}'

                    line += f'&{c} $\\pm$ {(rr - ll) / 2:.0e}'
                    if k == aulcs[2] - 1:
                        line += '\\\\\n'
                file.write(line)

            file.write('\\bottomrule\n')

from typing import Sequence, Tuple

import numpy as np


def hierarchical_bootstrap_ci(
    target_accuracies: Sequence[float],
    method_accuracies: Sequence[float],
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 100_000,
    n_subset_seeds: int = 3,
    random_state: int = 42,
) -> Tuple[float, float]:
    """
    Непарный иерархический bootstrap для разности точностей:

        delta = mean(target) - mean(method)

    На каждой bootstrap-итерации:

    1. Для target независимо ресемплируются subset seeds.
    2. Внутри каждого выбранного subset seed ресемплируются
       training seeds.
    3. То же независимо выполняется для сравниваемого метода.
    4. Сохраняется разность bootstrap-средних.

    Число subset seeds определяется автоматически отдельно для
    target и method. Предполагается, что для каждого subset seed
    выполнено ровно 5 training runs:

        15 значений = 3 subset seeds × 5 training seeds;
        10 значений = 2 subset seeds × 5 training seeds.

    Parameters
    ----------
    target_accuracies:
        Точности H&GradNorm. Массив длины 10 или 15.

    method_accuracies:
        Точности сравниваемого метода. Массив длины 10 или 15.

    confidence:
        Уровень доверия. Можно передавать как 0.95 или как 95.
        Аналогично поддерживаются 0.90/90 и 0.99/99.

    n_bootstrap:
        Число bootstrap-итераций.

    n_subset_seeds:
        Максимальное ожидаемое число subset seeds. Сохранено
        в сигнатуре для обратной совместимости. Обычно равно 3.

    random_state:
        Seed генератора случайных чисел.

    Returns
    -------
    ci_low, ci_high:
        Нижняя и верхняя границы percentile bootstrap CI.
    """

    target = np.asarray(
        target_accuracies,
        dtype=np.float64,
    ).reshape(-1)

    method = np.asarray(
        method_accuracies,
        dtype=np.float64,
    ).reshape(-1)

    if target.size == 0 or method.size == 0:
        raise ValueError("Input arrays must not be empty.")

    if not np.all(np.isfinite(target)):
        raise ValueError(
            "target_accuracies contains NaN or infinity."
        )

    if not np.all(np.isfinite(method)):
        raise ValueError(
            "method_accuracies contains NaN or infinity."
        )

    if n_subset_seeds < 2:
        raise ValueError(
            "n_subset_seeds must be at least 2."
        )

    if n_bootstrap <= 0:
        raise ValueError(
            "n_bootstrap must be positive."
        )

    # Поддерживаем confidence=95 и confidence=0.95.
    if confidence > 1:
        confidence = confidence / 100.0

    if not 0 < confidence < 1:
        raise ValueError(
            "confidence must be in (0, 1), or expressed "
            "as a percentage."
        )

    # В эксперименте на каждый subset seed приходится
    # ровно 5 независимых запусков обучения.
    n_train_seeds = 5

    def infer_n_subset_seeds(
        values: np.ndarray,
        array_name: str,
    ) -> int:
        if values.size % n_train_seeds != 0:
            raise ValueError(
                f"{array_name} size {values.size} is not "
                f"divisible by n_train_seeds={n_train_seeds}."
            )

        inferred = values.size // n_train_seeds

        if inferred < 2:
            raise ValueError(
                f"{array_name} contains only {inferred} subset "
                "seed. Hierarchical bootstrap requires at least 2."
            )

        if inferred > n_subset_seeds:
            raise ValueError(
                f"{array_name} contains {inferred} subset seeds, "
                f"which exceeds n_subset_seeds={n_subset_seeds}."
            )

        return inferred

    n_target_subset_seeds = infer_n_subset_seeds(
        target,
        "target",
    )

    n_method_subset_seeds = infer_n_subset_seeds(
        method,
        "method",
    )

    target = target.reshape(
        n_target_subset_seeds,
        n_train_seeds,
    )

    method = method.reshape(
        n_method_subset_seeds,
        n_train_seeds,
    )

    rng = np.random.default_rng(random_state)

    bootstrap_differences = np.empty(
        n_bootstrap,
        dtype=np.float64,
    )

    for bootstrap_idx in range(n_bootstrap):
        # =====================================================
        # Независимый bootstrap иерархии H&GradNorm
        # =====================================================

        target_subset_indices = rng.integers(
            low=0,
            high=n_target_subset_seeds,
            size=n_target_subset_seeds,
        )

        target_train_indices = rng.integers(
            low=0,
            high=n_train_seeds,
            size=(
                n_target_subset_seeds,
                n_train_seeds,
            ),
        )

        target_sample = target[
            target_subset_indices[:, None],
            target_train_indices,
        ]

        # =====================================================
        # Независимый bootstrap иерархии baseline
        # =====================================================

        method_subset_indices = rng.integers(
            low=0,
            high=n_method_subset_seeds,
            size=n_method_subset_seeds,
        )

        method_train_indices = rng.integers(
            low=0,
            high=n_train_seeds,
            size=(
                n_method_subset_seeds,
                n_train_seeds,
            ),
        )

        method_sample = method[
            method_subset_indices[:, None],
            method_train_indices,
        ]

        bootstrap_differences[bootstrap_idx] = (
            target_sample.mean()
            - method_sample.mean()
        )

    alpha = 1.0 - confidence

    ci_low, ci_high = np.quantile(
        bootstrap_differences,
        [
            alpha / 2.0,
            1.0 - alpha / 2.0,
        ],
    )

    return float(ci_low), float(ci_high)
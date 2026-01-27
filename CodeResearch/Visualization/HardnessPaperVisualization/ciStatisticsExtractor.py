import math

import numpy as np
from typing import Tuple, Optional

from CodeResearch.Visualization.HardnessPaperVisualization.extractData import extractTask, extractFilesForParameters
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
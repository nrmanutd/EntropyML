import math
from pathlib import Path

import numpy as np

from CodeResearch.Visualization.HardnessPaperVisualization.LatexConstants import constants
from CodeResearch.Visualization.HardnessPaperVisualization.Services.ciStatisticsExtractor import hierarchical_bootstrap_ci
from CodeResearch.Visualization.HardnessPaperVisualization.Services.extractData import getFileForConcreteTask
from CodeResearch.Visualization.saveDataForVisualization import deserialize_labeles_list_of_arrays

def extractConcreteCI(folder, task, fraction, method, targetMethod):
    print(f'{task}, {fraction}, {method} vs {targetMethod}')

    file = getFileForConcreteTask(folder, task, fraction, method)
    targetFile = getFileForConcreteTask(folder, task, fraction, targetMethod)
    targetFolder = f'{folder}/{task}_epoch'

    rr = deserialize_labeles_list_of_arrays(f'{targetFolder}/{file}')
    arrays = rr[0]

    rrTarget = deserialize_labeles_list_of_arrays(f'{targetFolder}/{targetFile}')
    arraysTarget = rrTarget[0]

    finalAccuracies = arrays[-1]
    finalAccuraciesTarget = arraysTarget[-1]

    ci = hierarchical_bootstrap_ci(finalAccuraciesTarget, finalAccuracies)

    print(f'95% ci: [{ci[0]}; {ci[1]}]')

    return ci

def extractConcreteStatistics(folder, task, fraction, method):
    print(f'{task}, {fraction}, {method}')

    file = getFileForConcreteTask(folder, task, fraction, method)
    targetFolder = f'{folder}/{task}_epoch'

    rr = deserialize_labeles_list_of_arrays(f'{targetFolder}/{file}')
    arrays = rr[0]

    finalAccuracies = arrays[-1]
    bestMean = np.mean(finalAccuracies)
    std = np.std(finalAccuracies) / math.sqrt(len(finalAccuracies))

    print(f'mean = {bestMean}, std = {std}')

    return bestMean, std

def estimateTopTwo(meanTable):
    s = meanTable.shape

    result = np.zeros((s[0], s[1], 3))
    for i in range(s[0]):
        for j in range(s[1]):
            bestAccuracy = None
            bestIdx = None

            secondBestAccuracy = None
            secondBestIdx = None

            maxIdx = s[2] - 1
            for k in range(s[2]):
                if bestIdx is None:
                    bestIdx = k
                    bestAccuracy = meanTable[i, j, k]
                    continue

                current = meanTable[i, j, k]
                if current > bestAccuracy:
                    bestAccuracy = current
                    bestIdx = k

            for k in range(s[2]):
                if k == bestIdx:
                    continue

                if secondBestIdx is None:
                    secondBestIdx = k
                    secondBestAccuracy = meanTable[i, j, k]
                    continue

                current = meanTable[i, j, k]
                if current > secondBestAccuracy:
                    secondBestAccuracy = current
                    secondBestIdx = k

            if bestIdx != maxIdx and math.fabs(meanTable[i, j, bestIdx] - meanTable[i, j, maxIdx]) < 0.01:
                result[i, j, 2] = maxIdx
            else:
                result[i, j, 2] = -1

            result[i, j, 0] = bestIdx
            result[i, j, 1] = secondBestIdx

    return result

def estimateDelta(meanTable):
    s = meanTable.shape

    result = np.zeros((s[0], s[1], s[2]))
    for i in range(s[0]):
        for j in range(s[1]):
            for k in range(s[2]):
                best = None
                for m in range(s[3] - 1):
                    if best is None:
                        best = meanTable[i, j, k, s[3] - 1] - meanTable[i, j, k, m]
                        continue

                    current = meanTable[i, j, k, s[3] - 1] - meanTable[i, j, k, m]
                    if best > current:
                        best = current

                result[i, j, k] = best

    return result

def getDatasetVisualizationName(task):
    if task == 'mnist':
        return 'MNIST'
    elif task == 'cifar':
        return 'CIFAR 10'
    elif task == 'cifar100':
        return 'CIFAR 100'
    elif task == 'svhn':
        return 'SVHN'
    else:
        raise ValueError(f'Incorrect dataset name: {task}')

def getMethodVisualizationName(method):
    if method == 'rand':
        return 'Random'
    elif method == 'EL2N':
        return 'EL2N'
    elif method == 'GradNorm':
        return 'GraND'
    elif method == 'chg_inc':
        return 'CHG'
    elif method == 'forgetting':
        return 'Forgetting'
    elif method == 'k-centered_inc':
        return '$k$-centered'
    elif method == 'boss':
        return 'BOSS'
    elif method == 'GradNorm_inc':
        return '$\\boldsymbol{\\mathrm{GraND}_{\\mathrm{inc}}}$'
    elif method == 'h':
        return 'h'
    elif method == 'h_inc':
        return '$\\boldsymbol{h_{\\mathrm{inc}}}$'
    else:
        raise ValueError(f'Incorrect method: {method}')

def fmt(x):
    s = f"{x:.1f}"
    return s

def saveRanksToFile(ranksTable, methods, fileName, tableBegin, tableEnd):
    file_path = Path(fileName)
    parent_dir = file_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    s = ranksTable.shape

    with open(fileName, 'w', encoding='utf=8') as file:
        file.write(tableBegin)
        file.write('\n')
        file.write('\\midrule\n')
        for i in range(s[0]):
            methodName = getMethodVisualizationName(methods[i])
            file.write(f'{methodName} & {ranksTable[i, 0]:.0f} & {ranksTable[i, 1]:.0f} & {ranksTable[i, 2]:.0f} \\\\\n')

        file.write(tableEnd)

def saveCIToFile(ciTable, tasks, fractions, fileName, tableBegin, tableEnd):
    file_path = Path(fileName)
    parent_dir = file_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    s = ciTable.shape

    with open(fileName, 'w', encoding='utf=8') as file:
        file.write(tableBegin)
        for i in range(s[0]):
            taskName = getDatasetVisualizationName(tasks[i])
            file.write('\\midrule\n')
            file.write('\\multirow[c]{4}{*}{\\textbf{' + taskName + '}}\n')
            for j in range(s[1]):
                file.write(f'& {int(float(fractions[j]) * 100)}\\%\n')

                for k in range(s[2]):
                    ci_left = ciTable[i, j, k, 0] * 100
                    ci_right = ciTable[i, j, k, 1] * 100

                    ci_left_str = fmt(ci_left)
                    ci_right_str = fmt(ci_right)

                    if ci_left > 0:
                        file.write('& $\\mathbf{' + f'[{ci_left_str}, {ci_right_str}]' + '}$\n')
                    elif ci_right > 0:
                        file.write(f'& $[{ci_left_str}, {ci_right_str}]$\n')
                    else:
                        file.write('& $\\underline{' + f'[{ci_left_str}, {ci_right_str}]' + '}$\n')

                file.write('\\\\\n')

        file.write(tableEnd)


def saveAccuraciesToFile(meanTable, bestDelta, stdTable, tasks, fractions, fileName, tableBegin, tableEnd):
    file_path = Path(fileName)
    parent_dir = file_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    s = meanTable.shape

    with open(fileName, 'w', encoding='utf=8') as file:
        file.write(tableBegin)
        file.write('\n')

        for i in range(s[0]):
            taskName = getDatasetVisualizationName(tasks[i])
            file.write('\\midrule\n')
            file.write('\\multirow[c]{4}{*}{\\textbf{' + taskName + '}}\n')
            for j in range(s[1]):
                v = fractions[j]
                file.write(f'& {int(float(fractions[j]) * 100)}\\%\n')

                for k in range(s[2]):
                    cur = meanTable[i, j, k] * 100
                    std = stdTable[i, j, k] * 100

                    dagger = '\\dagger' if k == bestDelta[i, j ,2] else ''
                    if tasks[i] == 'mnist' and (v == '0.2' or v == '0.5'):
                        dagger = ''

                    if k == bestDelta[i, j, 0]:
                        file.write('& $\\mathbf{' + f'{cur: .1f} \\pm {std: .1f}' + '}$\n')
                    elif k == bestDelta[i, j, 1]:
                        file.write('& $\\underline{' + f'{cur: .1f} \\pm {std: .1f}' + '}'+dagger + '$\n')
                    else:
                        file.write(f'& ${cur: .1f} \\pm {std: .1f}{dagger}$\n')

                file.write('\\\\\n')

        file.write(tableEnd)

def saveDataToFile(meanTable, bestDelta, stdTable, tasks, fractions,  fileName):
    s = meanTable.shape
    currentCount = 0

    with open(fileName, 'w', encoding='utf=8') as file:
        for i in range(s[0]):
            for j in range(s[1]):
                currentCount += 1

                if currentCount == 1:
                    file.write('\\multirow{12}{*}{\\rotatebox[origin=c]{90}{Protocol A: Hold-out}}\n')

                if currentCount == 4:
                    file.write('\\multirow{12}{*}{\\rotatebox[origin=c]{90}{Protocol B: Cross-split}}\n')

                line = '& \\multirow{4}{*}{\\rotatebox[origin=c]{90}{' + f'{tasks[j]}' + '}}'
                for k in range(s[2]):
                    line += f'& {fractions[k]} ' if k == 0 else f'& & {fractions[k]} '

                    if k == s[2] - 1:
                        cur = meanTable[i, j, k, 0]
                        std = stdTable[i, j, k, 0]
                        line += '& \\multicolumn{3}{c}{'+ f'{cur: .2g} $\\pm$ {std: .0e}' +'} \\\\\n'
                        if currentCount == 3:
                            line += '\\midrule\n'
                        elif currentCount == 6:
                            line += '\\bottomrule\n'
                        else:
                            line += '\\cmidrule(lr){2-6}\n'

                        continue

                    for m in range(s[3]):
                        if m == s[3] - 1:
                            cur = meanTable[i, j, k, m]
                            delta = bestDelta[i, j, k]
                            std = stdTable[i, j, k, m]

                            if delta >= 0:
                                curFormat = '\\textbf{' + f'{'+' if delta >= 0 else ''}{delta:.2f}' + '}'
                            else:
                                curFormat = f'{delta:.2f}'

                            line += f'& {cur:.2g}({curFormat}) $\\pm$ {std: .0e}\\\\\n'
                            continue

                        cur = meanTable[i, j, k, m]
                        std = stdTable[i, j, k, m]
                        line += f'& {cur:.2g} $\\pm$ {std: .0e} '

                file.write(line)
def saveAulcToFile(meanTable, bestDelta, stdTable, tasks, fractions,  fileName):
    s = meanTable.shape
    currentCount = 0

    with open(fileName, 'w', encoding='utf=8') as file:
        for i in range(s[0]):
            for j in range(s[1]):
                currentCount += 1

                if currentCount == 1:
                    file.write('\\multirow{12}{*}{\\rotatebox[origin=c]{90}{Protocol A: Hold-out}}\n')

                if currentCount == 4:
                    file.write('\\multirow{12}{*}{\\rotatebox[origin=c]{90}{Protocol B: Cross-split}}\n')

                line = '& \\multirow{4}{*}{\\rotatebox[origin=c]{90}{' + f'{tasks[j]}' + '}}'
                for k in range(s[2]):
                    line += f'& {fractions[k]} ' if k == 0 else f'& & {fractions[k]} '

                    if k == s[2] - 1:
                        cur = meanTable[i, j, k, 0]
                        std = stdTable[i, j, k, 0]
                        line += '& \\multicolumn{3}{c}{'+ f'{cur: .2g} $\\pm$ {std: .0e}' +'} \\\\\n'
                        if currentCount == 3:
                            line += '\\midrule\n'
                        elif currentCount == 6:
                            line += '\\bottomrule\n'
                        else:
                            line += '\\cmidrule(lr){2-6}\n'

                        continue

                    for m in range(s[3]):
                        if m == s[3] - 1:
                            cur = meanTable[i, j, k, m]
                            delta = bestDelta[i, j, k]
                            std = stdTable[i, j, k, m]

                            curFormat = '\\textbf{' + f'+{delta:.2f}' + '}'
                            line += f'& {cur:.2g}({curFormat}) $\\pm$ {std: .0e}\\\\\n'
                            continue

                        cur = meanTable[i, j, k, m]
                        std = stdTable[i, j, k, m]
                        line += f'& {cur:.2g} $\\pm$ {std: .0e} '

                file.write(line)
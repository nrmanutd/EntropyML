import math

import numpy as np
from CodeResearch.Visualization.HardnessPaperVisualization.extractData import extractTask, extractFilesForParameters
from CodeResearch.Visualization.saveDataForVisualization import deserialize_labeles_list_of_arrays


def extractConcreteStatistics(folder, protocol, task, fraction, mode):
    r = extractTask(folder, task.lower())

    files = extractFilesForParameters(r, fraction, protocol, mode)

    bestAccuracies = []
    for file in files:
        rr = deserialize_labeles_list_of_arrays(file['file'])
        arrays = rr[0]

        for i in range(len(arrays[0])):
            curMax = 0
            for j in range(len(arrays)):
                if arrays[j][i] > curMax:
                    curMax = arrays[j][i]
                    continue

            bestAccuracies.append(curMax)

    bestAccuracies = np.array(bestAccuracies)
    bestMean = np.mean(bestAccuracies)
    std = np.std(bestAccuracies) / math.sqrt(len(bestAccuracies))
    return bestMean, std



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
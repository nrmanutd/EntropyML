import math

import numpy as np

from CodeResearch.Visualization.HardnessPaperVisualization.Services.extractData import extractTask, extractFilesForParameters
from CodeResearch.Visualization.saveDataForVisualization import deserialize_labeles_list_of_arrays

def estimateAulcDelta(meanTable):
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

def extractAulc(folder, protocol, task, fraction, mode):
    r = extractTask(folder, task.lower())

    files = extractFilesForParameters(r, fraction, protocol, mode)

    areas = []
    for file in files:
        rr = deserialize_labeles_list_of_arrays(file['file'])
        arrays = rr[0]

        for i in range(len(arrays[0])):
            curArea = 0
            for j in range(len(arrays)):
                curArea += arrays[j][i] / len(arrays)

            areas.append(curArea)

    areas = np.array(areas)
    mean = np.mean(areas)
    std = np.std(areas) / math.sqrt(len(areas))
    return mean, std

def saveAulcToFile(meanTable, bestDelta, stdTable, tasks, fractions,  fileName):
    s = meanTable.shape

    for i in range(s[0]):
        currentCount = 0
        with open(f'{i}_{fileName}', 'w', encoding='utf=8') as file:
            for j in range(s[1]):
                currentCount += 1

                file.write('\\multirow{3}{*}{\\rotatebox[origin=c]{90}{\\scriptsize' + f' {tasks[j]}' + '}}\n')
                for k in range(s[2]):
                    line = f'& {fractions[k]} '

                    for m in range(s[3]):
                        if m == s[3] - 1:
                            cur = meanTable[i, j, k, m]
                            delta = bestDelta[i, j, k]
                            std = stdTable[i, j, k, m]

                            if delta > 0:
                                curFormat = '\\textbf{' + f'{'+' if delta > 0 else ''}{delta:.2f}' + '}'
                            else:
                                curFormat = f'{delta:.2f}'

                            line += f'& {cur:.2g}({curFormat}) $\\pm$ {std: .0e}\\\\\n'
                            continue

                        cur = meanTable[i, j, k, m]
                        std = stdTable[i, j, k, m]
                        line += f'& {cur:.2g} $\\pm$ {std: .0e} '

                    file.write(line)

                if currentCount == 3:
                    file.write('\\bottomrule\n')
                else:
                    file.write('\\midrule\n')
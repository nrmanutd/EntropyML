import numpy as np

from CodeResearch.Visualization.HardnessPaperVisualization.statisticsExtractor import extractConcreteStatistics, \
    estimateTopTwo, saveAccuraciesToFile, extractConcreteCI, saveCIToFile, saveRanksToFile


def evaluateAndSaveAccuracyTable(folder, task, fraction, method, fileName):
    meanTable = np.zeros((len(task), len(fraction), len(method)))
    stdTable = np.zeros((len(task), len(fraction), len(method)))

    for it in range(len(task)):
        for iif in range(len(fraction)):
            for im in range(len(method)):
                t = task[it]
                f = fraction[iif]
                m = method[im]

                best, std = extractConcreteStatistics(folder, t, f, m)
                meanTable[it, iif, im] = best
                stdTable[it, iif, im] = std

    topMethods = np.zeros((len(task), len(fraction), 2))  # top 2

    bestDelta = estimateTopTwo(meanTable)
    saveAccuraciesToFile(meanTable, bestDelta, stdTable, task, fraction, fileName)

def evaluateAndSaveCITableAndRankingsTable(folder, task, fraction, method, targetMethod, fileName, ranksFileName):
    ciTable = np.zeros((len(task), len(fraction), len(method), 2))
    ranksTable = np.zeros((len(method), 3))

    for it in range(len(task)):
        for iif in range(len(fraction)):
            for im in range(len(method)):
                t = task[it]
                f = fraction[iif]
                m = method[im]

                ci = extractConcreteCI(folder, t, f, m, targetMethod)
                ciTable[it, iif, im, 0] = ci[0]
                ciTable[it, iif, im, 1] = ci[1]

                if (ci[0] < 0 and ci[1] > 0):
                    idx = 1
                elif ci[1] < 0:
                    idx = 2
                else:
                    idx = 0

                ranksTable[im, idx] += 1

    saveCIToFile(ciTable, task, fraction, fileName)
    saveRanksToFile(ranksTable, method, ranksFileName)
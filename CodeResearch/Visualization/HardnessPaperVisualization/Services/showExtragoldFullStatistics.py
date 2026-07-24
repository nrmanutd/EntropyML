import numpy as np

from CodeResearch.Visualization.HardnessPaperVisualization.Services.statisticsExtractor import extractConcreteStatistics, \
    estimateDelta, saveDataToFile

folder = '..\\..\\CurriculumLearning\\extragold'
task = ['CIFAR 10']
fraction = ['0.1', '0.2', '0.5', '1']
protocol = ['fixed test']
mode = ['l', 'h&h_inc', 'i', 'i_cos', 'i_inner_p', 'h&i_inc']

meanTable = np.zeros((len(protocol), len(task), len(fraction), len(mode)))
stdTable = np.zeros((len(protocol), len(task), len(fraction), len(mode)))

for ip in range(len(protocol)):
    for it in range(len(task)):
        for iif in range(len(fraction)):
            for im in range(len(mode)):
                p = protocol[ip]
                t = task[it]
                f = fraction[iif]
                m = mode[im]

                if f == '1' and m != 'l':
                    meanTable[ip, it, iif, im] = meanTable[ip, it, iif, 0]
                    stdTable[ip, it, iif, im] = stdTable[ip, it, iif, 0]
                    continue

                best, std = extractConcreteStatistics(folder, p, t, f, m)
                print(f'{p}, {t}, {f}, {m}')
                print(best)
                print(std)
                meanTable[ip, it, iif, im] = best
                stdTable[ip, it, iif, im] = std

bestDelta = estimateDelta(meanTable)
saveDataToFile(meanTable, bestDelta, stdTable, task, fraction, 'extraGoldStatistics.txt')

import numpy as np

from CodeResearch.Visualization.HardnessPaperVisualization.aulcStatisticsExtractor import extractAulc, saveAulcToFile, \
    estimateAulcDelta

folder = '..\\..\\CurriculumLearning\\gold'
task = ['MNIST', 'CIFAR 10', 'CIFAR 100']
fraction = ['0.1', '0.2', '0.5']
protocol = ['fixed test', 'random subset']
mode = ['l', 'h&h_inc', 'h&i_inc']

aulcTable = np.zeros((len(protocol), len(task), len(fraction), len(mode)))
stdTable = np.zeros((len(protocol), len(task), len(fraction), len(mode)))

for ip in range(len(protocol)):
    for it in range(len(task)):
        for iif in range(len(fraction)):
            for im in range(len(mode)):
                p = protocol[ip]
                t = task[it]
                f = fraction[iif]
                m = mode[im]

                aulc, std = extractAulc(folder, p, t, f, m)
                print(f'{p}, {t}, {f}, {m}')
                print(aulc)
                print(std)
                aulcTable[ip, it, iif, im] = aulc
                stdTable[ip, it, iif, im] = std

bestDelta = estimateAulcDelta(aulcTable)
saveAulcToFile(aulcTable, bestDelta, stdTable, task, fraction, 'aulc_statistics.txt')

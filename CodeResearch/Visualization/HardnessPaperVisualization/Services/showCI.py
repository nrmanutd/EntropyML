import numpy as np

from CodeResearch.Visualization.HardnessPaperVisualization.Services.ciStatisticsExtractor import extractAulcCI, extractAccCI, \
    saveCIToFile

folder = '..\\..\\CurriculumLearning\\gold'
task = ['MNIST', 'CIFAR 10', 'CIFAR 100']
fraction = ['0.1', '0.2', '0.5']
protocol = ['fixed test', 'random subset']

accTable = np.zeros((len(protocol), len(task), len(fraction), 3))
aulcTable = np.zeros((len(protocol), len(task), len(fraction), 3))

for ip in range(len(protocol)):
    for it in range(len(task)):
        for iif in range(len(fraction)):
            p = protocol[ip]
            t = task[it]
            f = fraction[iif]

            aulc, left, right = extractAulcCI(folder, p, t, f, 0.95)

            aulcTable[ip, it, iif, 0] = aulc
            aulcTable[ip, it, iif, 1] = left
            aulcTable[ip, it, iif, 2] = right

            acc, left, right = extractAccCI(folder, p, t, f, 0.95)

            accTable[ip, it, iif, 0] = acc
            accTable[ip, it, iif, 1] = left
            accTable[ip, it, iif, 2] = right

saveCIToFile(accTable, aulcTable, task, 'ci.txt')
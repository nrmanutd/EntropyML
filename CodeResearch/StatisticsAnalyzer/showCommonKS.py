import numpy as np
from scipy.stats import kendalltau, spearmanr, mannwhitneyu, linregress

from CodeResearch.StatisticsAnalyzer.dataLoader import loadKSData, loadNNData, loadPermutationData, loadKSsyncData, \
    loadNNsyncData
from CodeResearch.Visualization.VisualizeAndSaveCommonTopSubsamples import \
    visualizeAndSaveKSForEachPairAndTwoDistributions

taskName = 'mnist'
#taskName = 'cifar'

ksData = loadKSData(taskName)
nnData = loadNNData(taskName)

labels = ksData[1]

visualizeAndSaveKSForEachPairAndTwoDistributions(ksData[0], nnData[0], labels, taskName, 'total', folder='..\\PValuesFigures')
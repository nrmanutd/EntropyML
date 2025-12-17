import math
import time

import numpy as np

from CodeResearch.CurriculumLearning.clHelpers import filterDataSet, visualizeAndSaveComplexity, \
    plot_complexity_distributions, plot_distributions_kde_with_metrics, createLearnerBasedHardnessCalculator
from CodeResearch.DataSeparationFramework.Metrics.KSMetric import KSMetric
from CodeResearch.DataSeparationFramework.pValueCalculator import PValueCalculator
from CodeResearch.Helpers.Logger import SimpleLogger
from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger
from CodeResearch.LearningFramework.Learners.CompositeLearner import CompositeLearner
from CodeResearch.LearningFramework.Learners.NNLearner import NNLearner
from CodeResearch.LearningFramework.generalLearningEstimator import GeneralLearningEstimator
from CodeResearch.ObjectComplexity.Factory.ShapValuesComplexityCalculatorFactory import \
    ShapValuesComplexityCalculatorFactory
from CodeResearch.dataSets import loadCifar, loadMnist, load_proteins, make_spirals, loadCifar100

nAttempts = 100
nSamples = 2000
#x, y = make_random(nSamples)
#x, y = datasets.make_blobs(n_samples=nSamples, centers=2, n_features=2, random_state=42)
#x, y = make_xor(nSamples)
#x, y = datasets.make_circles(n_samples=nSamples, factor=0.5, noise=0.1, random_state=42)
#x, y = make_spirals(nSamples)
#x, y = loadMnist()
#x, y = loadCifar()
x, y = loadCifar100()
x, y = filterDataSet(x, y, 0.5, 43, 88)
#x, y = load_proteins("../Data/Proteins/df_master.csv")

logger = SimpleLogger.SimpleLogger()

taskName = 'cifar100_epoch'
fractions = [0.5]
hidden_sizes = (64, 64)
hardnessEpochs = 20

t = time.time()
for fraction in fractions:

    logger.logDebug(f'Calculating for fraction {fraction}, attempts: {nAttempts}')
    hc = createLearnerBasedHardnessCalculator(nAttempts, fraction, logger, x.shape[1], 2, hardnessEpochs,
                                              hidden_sizes)

    hardnessResult = hc.calculateHardness(x, y)
    importance = hardnessResult[0]
    easyness = hardnessResult[1]

    prefix = f'{taskName}\\{taskName}_{fraction}_{nAttempts}_{fraction}'
    plot_distributions_kde_with_metrics(easyness, importance, f'{prefix}_distribution.png')
    visualizeAndSaveComplexity(easyness, importance, f'{prefix}_complexity.png')

    easyThresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    for easy in easyThresholds:
        easyIdx = np.where(easyness > easy)[0]
        plot_distributions_kde_with_metrics(easyness[easyIdx], importance[easyIdx], f'{prefix}_distribution_conditioned_easy_{easy}.png')

    importantIdx = np.where(importance > 0)[0]
    plot_distributions_kde_with_metrics(easyness[importantIdx], importance[importantIdx], f'{prefix}_distribution_conditioned_important.png')

    print(f'Visualization for {fraction}, time: {time.time() - t} s')
import math
import time

import numpy as np

from CodeResearch.CurriculumLearning.clHelpers import filterDataSet, visualizeAndSaveComplexity, \
    plot_complexity_distributions, plot_distributions_kde_with_metrics
from CodeResearch.DataSeparationFramework.Metrics.KSMetric import KSMetric
from CodeResearch.DataSeparationFramework.pValueCalculator import PValueCalculator
from CodeResearch.LearningFramework.Learners.CompositeLearner import CompositeLearner
from CodeResearch.LearningFramework.Learners.NNLearner import NNLearner
from CodeResearch.LearningFramework.generalLearningEstimator import GeneralLearningEstimator
from CodeResearch.ObjectComplexity.Factory.ShapValuesComplexityCalculatorFactory import \
    ShapValuesComplexityCalculatorFactory
from CodeResearch.dataSets import loadCifar, loadMnist, load_proteins, make_spirals

nAttempts = 50
nSamples = 2000
#x, y = make_random(nSamples)
#x, y = datasets.make_blobs(n_samples=nSamples, centers=2, n_features=2, random_state=42)
#x, y = make_xor(nSamples)
#x, y = datasets.make_circles(n_samples=nSamples, factor=0.5, noise=0.1, random_state=42)
#x, y = make_spirals(nSamples)
#x, y = loadMnist()
#x, y = loadCifar()
#x, y = filterDataSet(x, y, 1, 3, 5)
x, y = load_proteins("../Data/Proteins/df_master.csv")

x = np.hstack((x, -x))

taskName = 'spirals'
fractions = [0.5]

targetProduct = 0.2 * 100 * 5

t = time.time()
for fraction in fractions:
    currentAttempts = math.ceil(targetProduct / fraction)

    print(f'Calculating for fraction {fraction}, attempts: {currentAttempts}')

    hardnessCalculator = PValueCalculator(ShapValuesComplexityCalculatorFactory(), KSMetric(), currentAttempts, True, False,
                                          False)
    result = hardnessCalculator.calcPValueFastPro(math.ceil(len(y) * fraction), x, y, 0, 1)
    complexityCalculator = result[2]
    importance, easiness = complexityCalculator.getShapValues()

    prefix = f'{taskName}\\{taskName}_{fraction}_{currentAttempts}_{fraction}'
    plot_distributions_kde_with_metrics(easiness, importance, f'{prefix}_distribution.png')

    easyThresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    for easy in easyThresholds:
        easyIdx = np.where(easiness > easy)[0]
        plot_distributions_kde_with_metrics(easiness[easyIdx], importance[easyIdx], f'{prefix}_distribution_conditioned_easy_{easy}.png')

    importantIdx = np.where(importance > 0)[0]
    plot_distributions_kde_with_metrics(easiness[importantIdx], importance[importantIdx], f'{prefix}_distribution_conditioned_important.png')

    visualizeAndSaveComplexity(easiness, importance, f'{prefix}_complexity.png')

    print(f'Visualization for {fraction}, time: {time.time() - t} s')
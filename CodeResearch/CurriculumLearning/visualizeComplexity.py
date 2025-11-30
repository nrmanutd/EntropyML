import math

import numpy as np

from CodeResearch.CurriculumLearning.clHelpers import filterDataSet, visualizeAndSaveComplexity, \
    plot_complexity_distributions
from CodeResearch.DataSeparationFramework.Metrics.KSMetric import KSMetric
from CodeResearch.DataSeparationFramework.pValueCalculator import PValueCalculator
from CodeResearch.LearningFramework.Learners.CompositeLearner import CompositeLearner
from CodeResearch.LearningFramework.Learners.NNLearner import NNLearner
from CodeResearch.LearningFramework.generalLearningEstimator import GeneralLearningEstimator
from CodeResearch.ObjectComplexity.Factory.ShapValuesComplexityCalculatorFactory import \
    ShapValuesComplexityCalculatorFactory
from CodeResearch.dataSets import loadCifar

nAttempts = 1000
nSamples = 2000
#x, y = make_random(nSamples)
#x, y = datasets.make_blobs(n_samples=nSamples, centers=2, n_features=2, random_state=42)
#x, y = make_xor(nSamples)
#x, y = datasets.make_circles(n_samples=nSamples, factor=0.5, noise=0.1, random_state=42)
#x, y = make_spirals(nSamples)
#x, y = loadMnist()
x, y = loadCifar()
x, y = filterDataSet(x, y, 1, 3, 5)

x = np.hstack((x, -x))

taskName = 'cifar'
fraction = 0.1

alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

hardnessCalculator = PValueCalculator(ShapValuesComplexityCalculatorFactory(), KSMetric(), nAttempts,  True, False, False)
result = hardnessCalculator.calcPValueFastPro(math.ceil(len(y) * fraction), x, y, 0, 1)
complexityCalculator = result[2]
importance, easiness = complexityCalculator.getShapValues()

prefix = f'{taskName}_{fraction}_{nAttempts}'
plot_complexity_distributions(easiness, importance, f'{prefix}_distribution.png')

easyIdx = np.where(easiness > 0.5)[0]
plot_complexity_distributions(easiness[easyIdx], importance[easyIdx], f'{prefix}_distribution_conditioned_easy.png')

importantIdx = np.where(importance > 0)[0]
plot_complexity_distributions(easiness[importantIdx], importance[importantIdx], f'{prefix}_distribution_conditioned_important.png')

visualizeAndSaveComplexity(easiness, importance, f'{prefix}_complexity.png')
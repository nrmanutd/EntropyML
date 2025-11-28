import time

from CodeResearch.CurriculumLearning.clHelpers import calculateLosses
from CodeResearch.LearningFramework.Learners.XGBoostLearner import XGBoostLearner
from CodeResearch.LearningFramework.generalLearningEstimator import GeneralLearningEstimator
from CodeResearch.Visualization.visualizeLearningErrors import plot_multi_errors_vs_alpha
from CodeResearch.dataSets import make_xor

nIterations = 100
nAttempts = 100
nSamples = 2000
#x, y = make_random(nSamples)
#x, y = datasets.make_blobs(n_samples=nSamples, centers=2, n_features=2, random_state=42)
x, y = make_xor(nSamples)

taskName = 'xor'

alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
xgbLearner = XGBoostLearner()
generalLearner = GeneralLearningEstimator(nIterations)

losses = []
lossesHardness = []
lossesImportant = []
lossesHardAndImportant = []

t = time.time()
for i in range(len(alphas)):
    alpha = alphas[i]
    print('################## Alpha = ', alpha)

    #curLosses = calculateLosses(x, y, alpha, 0.5, nAttempts, False, False, generalLearner, xgbLearner)
    #curLossesHardness = calculateLosses(x, y, alpha, 0.5, nAttempts, False, True, generalLearner, xgbLearner)
    curLossesImportance = calculateLosses(x, y, alpha, 0.5, nAttempts, True, False, generalLearner, xgbLearner)
    curLossesHardnessAndImportance = calculateLosses(x, y, alpha, 0.5, nAttempts, True, True, generalLearner,
                                                     xgbLearner)

    curLosses = curLossesImportance
    curLossesHardness =curLossesImportance

    losses.append(curLosses)
    lossesHardness.append(curLossesHardness)
    lossesImportant.append(curLossesImportance)
    lossesHardAndImportant.append(curLossesHardnessAndImportance)

    plot_multi_errors_vs_alpha([losses, lossesHardness, lossesImportant, lossesHardAndImportant], alphas[:(i + 1)],
                               ['l', 'hard', 'important', 'both'], taskName, f'{taskName}_{nIterations}')

    print('######################## Current time: ' + str(time.time() - t))

plot_multi_errors_vs_alpha([losses, lossesHardness, lossesImportant, lossesHardAndImportant], alphas,
                           ['l', 'hard', 'important', 'both'], taskName, f'{taskName}_{nIterations}')

import time

from sklearn import datasets

from CodeResearch.CurriculumLearning.clHelpers import calculateLosses, filterDataSet
from CodeResearch.LearningFramework.Learners.CompositeLearner import CompositeLearner
from CodeResearch.LearningFramework.Learners.NNLearner import NNLearner
from CodeResearch.LearningFramework.Learners.XGBoostLearner import XGBoostLearner
from CodeResearch.LearningFramework.generalLearningEstimator import GeneralLearningEstimator
from CodeResearch.Visualization.visualizeLearningErrors import plot_multi_errors_vs_alpha
from CodeResearch.dataSets import make_xor, make_spirals, loadMnist, loadCifar

nIterations = 10
nAttempts = 100
nSamples = 2000
#x, y = make_random(nSamples)
#x, y = datasets.make_blobs(n_samples=nSamples, centers=2, n_features=2, random_state=42)
#x, y = make_xor(nSamples)
#x, y = datasets.make_circles(n_samples=nSamples, factor=0.5, noise=0.1, random_state=42)
#x, y = make_spirals(nSamples)
x, y = loadCifar()
x, y = filterDataSet(x, y, 0.2, 3, 5)

taskName = 'cifar'

alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

compositeLearner = CompositeLearner(NNLearner())
generalLearner = GeneralLearningEstimator(nIterations)

losses = []
lossesHardness = []
lossesImportant = []
lossesHardAndImportant = []

t = time.time()
for i in range(len(alphas)):
    alpha = alphas[i]
    print('################## Alpha = ', alpha)

    tripleLosses = calculateLosses(x, y, alpha, 0.5, nAttempts, generalLearner, compositeLearner)

    losses.append(tripleLosses[0])
    lossesImportant.append(tripleLosses[1])
    lossesHardness.append(tripleLosses[2])
    lossesHardAndImportant.append(tripleLosses[3])

    plot_multi_errors_vs_alpha([losses, lossesHardness, lossesImportant, lossesHardAndImportant], alphas[:(i + 1)],
                               ['l', 'hard', 'important', 'both'], taskName, f'{taskName}_{nIterations}')

    print('######################## Current time: ' + str(time.time() - t))

plot_multi_errors_vs_alpha([losses, lossesHardness, lossesImportant, lossesHardAndImportant], alphas,
                           ['l', 'hard', 'important', 'both'], taskName, f'{taskName}_{nIterations}')

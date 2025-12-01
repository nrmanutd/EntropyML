import time
import numpy as np

from CodeResearch.CurriculumLearning.clHelpers import calculateLosses, filterDataSet
from CodeResearch.LearningFramework.Learners.CompositeLearner import CompositeLearner
from CodeResearch.LearningFramework.Learners.NNLearner import NNLearner
from CodeResearch.LearningFramework.Learners.XGBoostLearner import XGBoostLearner
from CodeResearch.LearningFramework.generalLearningEstimator import GeneralLearningEstimator
from CodeResearch.Visualization.visualizeLearningErrors import plot_multi_errors_vs_alpha
from CodeResearch.dataSets import loadCifar, loadMnist, load_proteins

nIterations = 10
nAttempts = 200
nSamples = 2000
#x, y = make_random(nSamples)
#x, y = datasets.make_blobs(n_samples=nSamples, centers=2, n_features=2, random_state=42)
#x, y = make_xor(nSamples)
#x, y = datasets.make_circles(n_samples=nSamples, factor=0.5, noise=0.1, random_state=42)
#x, y = make_spirals(nSamples)
x, y = loadMnist()
#x, y = loadCifar()
x, y = filterDataSet(x, y, 1, 3, 5)
#x, y = load_proteins("../Data/Proteins/df_master.csv")

taskName = 'mnist'

alphas = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
fraction = 0.5
testAlpha = 0.5

compositeLearner = CompositeLearner(NNLearner())
generalLearner = GeneralLearningEstimator(nIterations)

losses = []
lossesHardness = []
lossesImportant = []
lossesHardAndImportant = []

t = time.time()

tripleLosses = calculateLosses(x, y, alphas / (1 - testAlpha), testAlpha, nAttempts, fraction, generalLearner, compositeLearner)

for i in range(len(alphas)):
    curShift = i*4
    losses.append(tripleLosses[curShift])
    lossesImportant.append(tripleLosses[curShift + 1])
    lossesHardness.append(tripleLosses[curShift + 2])
    lossesHardAndImportant.append(tripleLosses[curShift + 3])

plot_multi_errors_vs_alpha([losses, lossesHardness, lossesImportant, lossesHardAndImportant], alphas,
                               ['l', 'hard', 'important', 'both'], taskName, f'{taskName}_{nIterations}_{nAttempts}_{fraction}_NN')

print('######################## Current time: ' + str(time.time() - t))
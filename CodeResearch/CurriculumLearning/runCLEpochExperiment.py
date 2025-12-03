import time

import numpy as np

from CodeResearch.CurriculumLearning.clHelpers import calculateLosses, filterDataSet
from CodeResearch.LearningFramework.Learners.CompositeLearner import CompositeLearner
from CodeResearch.LearningFramework.Learners.NNPytorchEpochLearner import NNEpochLearnerPyTorch
from CodeResearch.LearningFramework.Learners.epochLearner import EpochLearner
from CodeResearch.LearningFramework.Loggers.EpochLearnerLogger import EpochLearnerLogger
from CodeResearch.LearningFramework.Samplers.SamplersFactories.RandomAllSetSamplerFactory import \
    RandomAllsetSamplerFactory
from CodeResearch.LearningFramework.generalLearningEstimator import GeneralLearningEstimator
from CodeResearch.ObjectComplexity.InstancePriority.PrioritizerType import PrioritizerType
from CodeResearch.dataSets import loadCifar, loadMnist, load_proteins, loadFashionMnist, loadCifar100

nIterations = 10
nAttempts = 200
nSamples = 2000
datasetFraction = 1
alphas = np.array([0.5])
fraction = 0.5
testAlpha = 0.5
epochs = 15
repeats = 20
nArrays = 8

#x, y = make_random(nSamples)
#x, y = datasets.make_blobs(n_samples=nSamples, centers=2, n_features=2, random_state=42)
#x, y = make_xor(nSamples)
#x, y = datasets.make_circles(n_samples=nSamples, factor=0.5, noise=0.1, random_state=42)
#x, y = make_spirals(nSamples)
#x, y = loadMnist()
#x, y = loadCifar()
#x, y = loadFashionMnist()
#x, y = filterDataSet(x, y, datasetFraction, firstClass, secondClass)
#x, y = load_proteins("../Data/Proteins/df_master.csv")

taskNames = ['cifar100_epoch', 'cifar100_epoch', 'cifar100_epoch', 'mnist_epoch','mnist_epoch', 'fashionMnist_epoch', 'fashionMnist_epoch', 'cifar_epoch', 'cifar_epoch', 'proteins_epoch']
firstClasses = [82, 13, 43, 5, 0, 0, 1, 3, 0, 0]
secondClasses = [62, 30, 88, 6, 1, 6, 5, 5, 8, 1]

for i in range(0, len(taskNames)):
    taskName = taskNames[i]
    firstClass = firstClasses[i]
    secondClass = secondClasses[i]

    if taskName == taskNames[0]:
        x, y = loadCifar100()
    elif taskName == taskNames[3]:
        x, y = loadMnist()
    elif taskName == taskNames[4]:
        x, y = loadFashionMnist()
    elif taskName == taskNames[7]:
        x, y = loadCifar()
    else:
        x, y = load_proteins("../Data/Proteins/df_master.csv")

    x, y = filterDataSet(x, y, datasetFraction, firstClass, secondClass)
    prefix = f'{taskName}_{nIterations}_{nAttempts}_{fraction}_{datasetFraction}_{repeats}_{nArrays}_{firstClass}_{secondClass}_NN'
    compositeLearner = CompositeLearner(EpochLearner(epochs, NNEpochLearnerPyTorch(),  RandomAllsetSamplerFactory(50, PrioritizerType.Probability)))
    logger = EpochLearnerLogger(epochs, taskName, prefix, nAttempts, repeats, nArrays)
    generalLearner = GeneralLearningEstimator(nIterations, logger)

    losses = []
    lossesHardness = []
    lossesImportant = []
    lossesHardAndImportant = []

    t = time.time()

    tripleLosses = calculateLosses(x, y, alphas / (1 - testAlpha), testAlpha, nAttempts, fraction, repeats, generalLearner, compositeLearner)
    print('######################## Current time: ' + str(time.time() - t))
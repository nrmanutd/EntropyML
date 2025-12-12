import numpy as np

from CodeResearch.CurriculumLearning.clHelpers import filterDataSet, \
    createLearnerBasedHardnessCalculator, createSampler
from CodeResearch.LearningFramework.Learners.CompositeLearner import CompositeLearner
from CodeResearch.LearningFramework.Learners.NNPytorchEpochLearner import NNEpochLearnerPyTorch
from CodeResearch.LearningFramework.Learners.epochLearner import EpochLearner
from CodeResearch.LearningFramework.Loggers.EpochLearnerLogger import EpochLearnerLogger
from CodeResearch.LearningFramework.Samplers.SamplersFactories.RandomAllSetSamplerFactory import \
    RandomAllsetSamplerFactory
from CodeResearch.LearningFramework.generalLearningEstimator import GeneralLearningEstimator
from CodeResearch.ObjectComplexity.InstancePriority.PrioritizerType import PrioritizerType
from CodeResearch.dataSets import loadCifar, loadMnist, loadCifar100

nIterations = 20
nAttempts = 100
nSamples = 2000
datasetFraction = 1
alphas = np.array([0.5])

fraction = 0.5
testAlpha = 0.5
epochs = 60

#best - 20 и (16, 16)
hardnessEpochs = 5
hidden_sizes = (8, 8)

repeats = 20
betas = [0.05, 0.1, 0.2, 0.5]
nArrays = 4 * len(betas)
batchSize = 50

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

taskNames = ['cifar100_epoch', 'cifar100_epoch', 'cifar100_epoch', 'cifar100_epoch', 'cifar100_epoch', 'cifar100_epoch', 'cifar100_epoch', 'mnist_epoch', 'cifar_epoch', 'cifar_epoch']
firstClasses = [43, 47, 43, 70, 9, 23, 5, 3, 0]
secondClasses = [88, 52, 87, 91, 10, 33, 6, 5, 8]

for i in range(1, len(taskNames)):
    taskName = taskNames[i]
    firstClass = firstClasses[i]
    secondClass = secondClasses[i]

    if taskName == taskNames[0]:
        x, y = loadCifar100()
    elif taskName == taskNames[7]:
        x, y = loadMnist()
    else:
        x, y = loadCifar()

    x, y = filterDataSet(x, y, datasetFraction, firstClass, secondClass)
    prefix = f'{taskName}_{nIterations}_{nAttempts}_{fraction}_{datasetFraction}_{repeats}_{nArrays}_{epochs}_{firstClass}_{secondClass}_NN'
    logger = EpochLearnerLogger(epochs, taskName, prefix, nAttempts, repeats, nArrays, betas)
    compositeLearner = CompositeLearner(EpochLearner(epochs, NNEpochLearnerPyTorch(),  RandomAllsetSamplerFactory(batchSize, PrioritizerType.Probability)), logger)
    generalLearner = GeneralLearningEstimator(nIterations, logger)

    hc = createLearnerBasedHardnessCalculator(nAttempts, fraction, logger, x.shape[1], len(np.unique(y)), hardnessEpochs, hidden_sizes)
    sampler = createSampler(x, y, alphas / (1 - testAlpha), betas, testAlpha, repeats, hc)

    logger.logDebug(f'Starting task {prefix}')
    generalLearner.estimateLearner(sampler, compositeLearner)
    logger.logDebug(f'Finished task {prefix}')
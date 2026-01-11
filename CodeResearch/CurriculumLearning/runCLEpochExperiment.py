import numpy as np

from CodeResearch.CurriculumLearning.clHelpers import filterDataSet, \
    createLearnerBasedHardnessCalculator, createSampler
from CodeResearch.Helpers.commonHelpers import normalizeTarget
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
nAttempts = 50
nSamples = 2000
datasetFraction = 1
alphas = np.array([0.5])

fraction = 0.5
testAlpha = 0.5
epochs = 60

#best - 20 и (16, 16)
hardnessEpochs = 20
hidden_sizes = (16, 16)

repeats = 10
betas = [0.05, 0.1, 0.2, 0.5, 1]
#betas = [0.05]

baseLabels = ['l', 'h&i_inc', 'h&h_inc']
nArrays = len(baseLabels) * len(betas)

batchSize = 50

dBatchSize = 50
dEpochs = 30
dHidden_sizes = (16, 16)
dAttempts = 100

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
secondClasses = [87, 52, 88, 91, 10, 33, 6, 5, 8]

for i in range(len(taskNames)):
    taskName = taskNames[i]
    firstClass = firstClasses[i]
    secondClass = secondClasses[i]

    if taskName == taskNames[0]:
        x, y = loadCifar100()
    elif taskName == taskNames[7]:
        x, y = loadMnist()
    else:
        x, y = loadCifar()

    y = normalizeTarget(y)

    x, y = filterDataSet(x, y, datasetFraction, firstClass, secondClass)
    y = normalizeTarget(y)
    nClasses = len(np.unique(y))

    prefix = f'{taskName}_{nIterations}_{nAttempts}_{fraction}_{datasetFraction}_{repeats}_{nArrays}_{epochs}_{firstClass}_{secondClass}_NN'
    logger = EpochLearnerLogger(epochs, taskName, prefix, nAttempts, repeats, nArrays, betas, baseLabels)
    compositeLearner = CompositeLearner(EpochLearner(epochs, NNEpochLearnerPyTorch(nClasses),  RandomAllsetSamplerFactory(batchSize, PrioritizerType.Probability)), logger)
    generalLearner = GeneralLearningEstimator(nIterations, logger)

    #hc = createLearnerBasedHardnessCalculator(nAttempts, logger, x.shape[1], nClasses, hardnessEpochs, hidden_sizes)
    sampler = createSampler(x, y, alphas / (1 - testAlpha), betas, testAlpha, repeats, lambda shouldUseLearner: createLearnerBasedHardnessCalculator(nAttempts, logger, x.shape[1], nClasses, hardnessEpochs, hidden_sizes, dAttempts, dBatchSize, dEpochs, dHidden_sizes, shouldUseLearner), logger)

    logger.logDebug(f'Starting task {prefix}')
    generalLearner.estimateLearner(sampler, compositeLearner)
    logger.logDebug(f'Finished task {prefix}')